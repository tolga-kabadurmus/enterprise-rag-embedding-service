"""Unit tests for the copied RustFS model-artifact repository."""

from pathlib import Path
import tempfile
import unittest

from app.common.settings.s3 import S3Settings
from app.repository.s3.boto3_repository import Boto3S3Repository


class FakePaginator:
    """Paginate two fake S3 response pages."""

    def __init__(self, pages: list[dict]) -> None:
        self.pages = pages
        self.parameters: dict[str, str] = {}

    def paginate(self, **parameters: str) -> list[dict]:
        self.parameters = parameters
        return self.pages


class FakeS3Client:
    """Minimal boto3-compatible client for repository tests."""

    def __init__(self, pages: list[dict]) -> None:
        self.paginator = FakePaginator(pages)
        self.downloads: list[tuple[str, str, str]] = []
        self.closed = False

    def get_paginator(self, name: str) -> FakePaginator:
        self.assert_name = name
        return self.paginator

    def download_file(self, bucket: str, key: str, destination: str, Config) -> None:
        self.downloads.append((bucket, key, destination))
        Path(destination).write_bytes(f"downloaded:{key}".encode())

    def close(self) -> None:
        self.closed = True


def settings() -> S3Settings:
    """Return safe test settings without reading the process environment."""

    return S3Settings(
        endpoint_url="http://rustfs:9000",
        access_key="test-access",
        secret_key="test-secret",
        verify_tls=False,
        max_concurrency=2,
    )


class S3RepositoryTests(unittest.IsolatedAsyncioTestCase):
    """Verify pagination, prefix mapping, traversal protection, and cleanup."""

    async def test_lists_all_pages(self) -> None:
        client = FakeS3Client(
            [
                {"Contents": [{"Key": "embeddings/a", "Size": 1}]},
                {"Contents": [{"Key": "embeddings/b", "Size": 2}]},
            ]
        )
        repository = Boto3S3Repository(settings(), client=client)

        objects = await repository.list_objects("models", "embeddings")

        self.assertEqual([item.key for item in objects], ["embeddings/a", "embeddings/b"])
        self.assertEqual(
            client.paginator.parameters,
            {"Bucket": "models", "Prefix": "embeddings/"},
        )
        await repository.close()
        self.assertTrue(client.closed)

    async def test_download_prefix_preserves_relative_paths(self) -> None:
        client = FakeS3Client(
            [
                {
                    "Contents": [
                        {"Key": "embeddings/intfloat/config.json", "Size": 1},
                        {"Key": "embeddings/intfloat/onnx/model.onnx", "Size": 1},
                    ]
                }
            ]
        )
        repository = Boto3S3Repository(settings(), client=client)

        with tempfile.TemporaryDirectory() as directory:
            paths = await repository.download_prefix(
                "models",
                "embeddings/intfloat",
                Path(directory),
            )
            self.assertEqual(
                sorted(path.relative_to(directory).as_posix() for path in paths),
                ["config.json", "onnx/model.onnx"],
            )
            self.assertEqual(
                (Path(directory) / "config.json").read_text(),
                "downloaded:embeddings/intfloat/config.json",
            )
        await repository.close()

    async def test_rejects_path_traversal(self) -> None:
        client = FakeS3Client(
            [{"Contents": [{"Key": "embeddings/intfloat/../secret", "Size": 1}]}]
        )
        repository = Boto3S3Repository(settings(), client=client)

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                await repository.download_prefix(
                    "models",
                    "embeddings/intfloat",
                    Path(directory),
                )
        await repository.close()


if __name__ == "__main__":
    unittest.main()
