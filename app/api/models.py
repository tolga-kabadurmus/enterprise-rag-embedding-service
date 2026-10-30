"""Pydantic models for the versioned embedding API."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


TextInput = Annotated[str, Field(min_length=1, max_length=32_000)]


class EmbeddingRequest(BaseModel):
    """Conceptual vectorization request for document or query text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_type: Literal["document", "query"]
    texts: list[TextInput] = Field(min_length=1, max_length=32)


class EmbeddingResponse(BaseModel):
    """Vector response shape reserved for the model implementation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    dimension: int = Field(ge=1)
    normalized: bool
    embeddings: list[list[float]]


class ErrorResponse(BaseModel):
    """Sanitized API error response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    detail: str = Field(min_length=1)
