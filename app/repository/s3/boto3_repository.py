"""RustFS-compatible S3 repository for model artifact synchronization."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import logging
from pathlib import Path
from typing import Any, TypeVar

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

from app.common.settings.s3 import S3Settings
from app.repository.models import S3ObjectMetadata
from app.repository.s3.base import StorageRepositoryBase

LOGGER = logging.getLogger(__name__)
ResultT = TypeVar("ResultT")


class S3Repository(StorageRepositoryBase):
    """Expose bounded asynchronous model-artifact operations over boto3.

    Boto3 remains blocking, so every operation crosses a bounded
    ``asyncio.to_thread`` boundary. Credentials are never included in logs.
    """

    def __init__(self, settings: S3Settings, client: Any | None = None) -> None:
        self._settings = settings
        self._client = client or boto3.client(
            "s3",
            endpoint_url=str(settings.endpoint_url),
            aws_access_key_id=settings.access_key,
            aws_secret_access_key=settings.secret_key.get_secret_value(),
            region_name=settings.region,
            verify=settings.verify_tls,
            config=Config(s3={"addressing_style": settings.addressing_style}),
        )
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)
        self._transfer_config = TransferConfig(use_threads=False)
        self._closed = False

    async def list_objects(self, bucket: str, key_prefix: str | None = None) -> list[S3ObjectMetadata]:
        """List all objects across every ``list_objects_v2`` page."""

        self._ensure_open()
        paginator = self._client.get_paginator("list_objects_v2")
        parameters: dict[str, str] = {"Bucket": bucket}
        if key_prefix:
            parameters["Prefix"] = key_prefix.strip("/") + "/"
        pages = await self._call(lambda: list(paginator.paginate(**parameters)))
        objects: list[S3ObjectMetadata] = []
        for page in pages:
            for item in page.get("Contents", []):
                objects.append(
                    S3ObjectMetadata(
                        bucket=bucket,
                        key=item["Key"],
                        size=item["Size"],
                        etag=self._normalize_etag(item.get("ETag")),
                        last_modified=item.get("LastModified"),
                    )
                )
        LOGGER.info("listed model objects bucket=%s key_prefix=%s count=%d", bucket, key_prefix, len(objects))
        return objects

    async def download_object(self, bucket: str, object_key: str, destination: Path) -> Path:
        """Download one object through a partial file and atomic replace."""

        self._ensure_open()
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(f".{destination.name}.part")
        try:
            await self._call(self._client.download_file, bucket, object_key, str(partial), Config=self._transfer_config)
            partial.replace(destination)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        LOGGER.info("downloaded model object bucket=%s object_key=%s destination=%s", bucket, object_key, destination)
        return destination

    async def download_prefix(self, bucket: str, key_prefix: str, destination_directory: Path) -> list[Path]:
        """Download every non-directory object under a prefix safely."""

        normalized_prefix = key_prefix.strip("/")
        if not normalized_prefix:
            raise ValueError("key_prefix must not be blank")
        object_prefix = f"{normalized_prefix}/"
        objects = await self.list_objects(bucket, normalized_prefix)
        downloads: list[tuple[str, Path]] = []
        for item in objects:
            if item.key.endswith("/"):
                continue
            relative_key = item.key.removeprefix(object_prefix)
            if relative_key == item.key:
                raise ValueError(f"object is outside prefix: {item.key}")
            downloads.append((item.key, self._safe_destination(destination_directory, relative_key)))

        paths: list[Path] = []
        batch_size = self._settings.max_concurrency
        for offset in range(0, len(downloads), batch_size):
            batch = downloads[offset : offset + batch_size]
            paths.extend(await asyncio.gather(*(self.download_object(bucket, key, path) for key, path in batch)))
        LOGGER.info("downloaded model prefix bucket=%s key_prefix=%s files=%d destination=%s", bucket, normalized_prefix, len(paths), destination_directory)
        return paths

    async def close(self) -> None:
        """Close the boto3 client once."""

        if self._closed:
            return
        async with self._semaphore:
            await asyncio.to_thread(self._client.close)
        self._closed = True
        LOGGER.info("embedding-service S3 repository closed")

    async def _call(self, operation: Callable[..., ResultT], *args: Any, **kwargs: Any) -> ResultT:
        self._ensure_open()
        async with self._semaphore:
            return await asyncio.to_thread(operation, *args, **kwargs)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("S3 repository is closed")

    @staticmethod
    def _safe_destination(root: Path, relative_key: str) -> Path:
        """Resolve a key under root and reject traversal or absolute paths."""

        candidate = (root / relative_key).resolve()
        resolved_root = root.resolve()
        if candidate != resolved_root and resolved_root not in candidate.parents:
            raise ValueError(f"unsafe model object key: {relative_key}")
        return candidate

    @staticmethod
    def _normalize_etag(value: Any) -> str | None:
        """Normalize quoted S3 ETags."""

        return None if value is None else str(value).strip('"')
