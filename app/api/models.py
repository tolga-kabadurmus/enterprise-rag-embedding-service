"""Pydantic models for the versioned embedding API."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.repository.embedding.base import SparseEmbeddingVector


TextInput = Annotated[str, Field(min_length=1, max_length=32_000)]


class EmbeddingRequest(BaseModel):
    """Conceptual vectorization request for document or query text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_type: Literal["document", "query"]
    texts: list[TextInput] = Field(min_length=1, max_length=32)


class EmbeddingResponse(BaseModel):
    """Dense and sparse vectors produced from the same ordered text batch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    dimension: int = Field(ge=1)
    normalized: bool
    embeddings: list[list[float]]
    sparse_model: str = Field(min_length=1)
    sparse_embeddings: list[SparseEmbeddingVector]

    @model_validator(mode="after")
    def validate_vector_batches(self) -> "EmbeddingResponse":
        """Require one correctly-sized dense and sparse vector per input."""

        if len(self.embeddings) != len(self.sparse_embeddings):
            raise ValueError("dense and sparse embedding counts must match")
        if any(len(vector) != self.dimension for vector in self.embeddings):
            raise ValueError("dense embedding dimension does not match metadata")
        return self


class ErrorResponse(BaseModel):
    """Sanitized API error response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    detail: str = Field(min_length=1)
