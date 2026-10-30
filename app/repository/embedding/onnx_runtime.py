"""ONNX Runtime implementation for multilingual E5 embeddings."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
import logging
from pathlib import Path
from typing import Any

import numpy as np

from app.common.settings.model import ModelSettings
from app.repository.embedding.base import EmbeddingRuntime

LOGGER = logging.getLogger(__name__)


class OnnxEmbeddingRuntime(EmbeddingRuntime):
    """Run E5 document/query embeddings with a loaded ONNX session.

    The runtime applies the E5 ``passage:`` or ``query:`` prefix, performs
    attention-mask-aware mean pooling, and L2-normalizes each vector. Native
    tokenizer and ONNX calls are executed on worker threads so the FastAPI
    event loop remains responsive.
    """

    def __init__(self, session: Any, tokenizer: Any, settings: ModelSettings) -> None:
        self._session = session
        self._tokenizer = tokenizer
        self._settings = settings

    @classmethod
    async def load(cls, settings: ModelSettings) -> "OnnxEmbeddingRuntime":
        """Load tokenizer and ONNX session from synchronized local artifacts."""

        if not settings.onnx_model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {settings.onnx_model_path}")
        if not settings.tokenizer_path.is_file():
            raise FileNotFoundError(f"tokenizer not found: {settings.tokenizer_path}")
        session, tokenizer = await asyncio.to_thread(
            cls._load_blocking,
            settings.onnx_model_path,
            settings.tokenizer_path,
        )
        runtime = cls(session, tokenizer, settings)
        LOGGER.info(
            "loaded embedding model model=%s revision=%s dimension=%d",
            settings.model_name,
            settings.revision,
            runtime.dimension,
        )
        return runtime

    @property
    def is_ready(self) -> bool:
        """Return whether both tokenizer and ONNX session are available."""

        return self._session is not None and self._tokenizer is not None

    @property
    def dimension(self) -> int:
        """Return the output vector dimension reported by the model."""

        outputs = self._session.get_outputs()
        shape = outputs[0].shape
        if len(shape) == 2 and isinstance(shape[-1], int):
            return shape[-1]
        return self._settings.dimension

    async def embed(
        self,
        texts: Sequence[str],
        input_type: str,
    ) -> list[list[float]]:
        """Generate normalized vectors for document or query inputs."""

        if input_type not in {"document", "query"}:
            raise ValueError("input_type must be document or query")
        if not texts:
            raise ValueError("texts must not be empty")
        prepared = [f"{self._prefix(input_type)}{text}" for text in texts]
        return await asyncio.to_thread(self._embed_blocking, prepared)

    async def close(self) -> None:
        """Release references to native model resources."""

        await asyncio.to_thread(self._close_blocking)

    def _embed_blocking(self, texts: list[str]) -> list[list[float]]:
        encodings = self._tokenizer.encode_batch(texts)
        input_ids = np.asarray([encoding.ids for encoding in encodings], dtype=np.int64)
        attention_mask = np.asarray(
            [encoding.attention_mask for encoding in encodings],
            dtype=np.int64,
        )
        inputs: dict[str, np.ndarray] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if any(item.name == "token_type_ids" for item in self._session.get_inputs()):
            inputs["token_type_ids"] = np.asarray(
                [encoding.type_ids for encoding in encodings],
                dtype=np.int64,
            )
        input_names = {item.name for item in self._session.get_inputs()}
        outputs = self._session.run(None, {key: value for key, value in inputs.items() if key in input_names})
        values = np.asarray(outputs[0])
        if values.ndim == 3:
            mask = attention_mask[..., np.newaxis].astype(values.dtype)
            values = (values * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        if values.ndim != 2:
            raise RuntimeError(f"unexpected embedding output shape: {values.shape}")
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        normalized = values / np.clip(norms, 1e-12, None)
        return normalized.astype(np.float32).tolist()

    @staticmethod
    def _load_blocking(model_path: Path, tokenizer_path: Path) -> tuple[Any, Any]:
        import onnxruntime
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        session = onnxruntime.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        return session, tokenizer

    @staticmethod
    def _prefix(input_type: str) -> str:
        return "passage: " if input_type == "document" else "query: "

    def _close_blocking(self) -> None:
        self._session = None
        self._tokenizer = None
