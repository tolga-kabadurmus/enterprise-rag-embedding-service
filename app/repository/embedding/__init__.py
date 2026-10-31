"""Embedding runtime contracts."""

from app.repository.embedding.base import EmbeddingRuntime, SparseEmbeddingVector
from app.repository.embedding.onnx_runtime import OnnxEmbeddingRuntime
from app.repository.embedding.scaffold import ScaffoldEmbeddingRuntime

__all__ = [
    "EmbeddingRuntime",
    "OnnxEmbeddingRuntime",
    "ScaffoldEmbeddingRuntime",
    "SparseEmbeddingVector",
]
