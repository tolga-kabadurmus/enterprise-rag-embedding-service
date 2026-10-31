"""Embedding runtime interface."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
import math

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SparseEmbeddingVector(BaseModel):
    """One validated sparse embedding with aligned token IDs and weights."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    indices: tuple[int, ...] = Field(
        description="Sorted, unique, non-negative token identifiers."
    )
    values: tuple[float, ...] = Field(
        description="Positive weights aligned by position with token identifiers."
    )

    @model_validator(mode="after")
    def validate_alignment(self) -> "SparseEmbeddingVector":
        """Reject empty, unaligned, or duplicate sparse-vector entries."""

        if not self.indices or len(self.indices) != len(self.values):
            raise ValueError("sparse indices and values must be non-empty and aligned")
        if len(set(self.indices)) != len(self.indices):
            raise ValueError("sparse indices must be unique")
        return self

    @field_validator("indices")
    @classmethod
    def validate_indices(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        """Require non-negative token IDs in deterministic ascending order."""

        if any(index < 0 for index in value):
            raise ValueError("sparse indices must be non-negative")
        if tuple(sorted(value)) != value:
            raise ValueError("sparse indices must be sorted")
        return value

    @field_validator("values")
    @classmethod
    def validate_values(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        """Require finite positive term-frequency weights."""

        if any(not math.isfinite(weight) or weight <= 0 for weight in value):
            raise ValueError("sparse values must be finite and positive")
        return value


class EmbeddingRuntime(ABC):
    """Lifecycle and inference contract for one loaded model runtime."""

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Return whether the runtime can serve inference requests."""

    @abstractmethod
    async def embed(
        self,
        texts: Sequence[str],
        input_type: str,
    ) -> list[list[float]]:
        """Vectorize text inputs once a real runtime is implemented."""

    @abstractmethod
    async def embed_sparse(
        self,
        texts: Sequence[str],
        input_type: str,
    ) -> list[SparseEmbeddingVector]:
        """Create token-frequency sparse vectors for document or query text."""

    @abstractmethod
    async def close(self) -> None:
        """Release model resources during application shutdown."""
