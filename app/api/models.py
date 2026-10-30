"""Pydantic models for the versioned embedding API."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.repository.embedding.base import SparseEmbeddingVector


TextInput = Annotated[
    str,
    Field(
        min_length=1,
        max_length=32_000,
        description="Non-empty source text bounded to 32,000 characters.",
    ),
]


class EmbeddingRequest(BaseModel):
    """Vectorization request for an ordered batch of document or query text."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "input_type": "query",
                    "texts": ["What is the annual leave policy?"],
                },
                {
                    "input_type": "document",
                    "texts": [
                        "Employees receive 20 days of annual leave.",
                        "Unused leave is handled according to local policy.",
                    ],
                },
            ]
        },
    )

    input_type: Literal["document", "query"] = Field(
        description=(
            "Preparation mode applied by the configured model. Document text "
            "uses passage preparation; questions use query preparation."
        ),
        examples=["query"],
    )
    texts: list[TextInput] = Field(
        min_length=1,
        max_length=32,
        description=(
            "Ordered text inputs. The response contains one dense and one "
            "sparse vector for each item in the same order."
        ),
    )


class EmbeddingResponse(BaseModel):
    """Dense and sparse vectors produced from the same ordered text batch."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "model": "intfloat/multilingual-e5-small",
                    "revision": "main",
                    "dimension": 3,
                    "normalized": True,
                    "embeddings": [[0.12, -0.34, 0.93]],
                    "sparse_model": "multilingual-e5-token-frequency-v1",
                    "sparse_embeddings": [
                        {"indices": [101, 2023], "values": [1.0, 1.0]}
                    ],
                }
            ]
        },
    )

    model: str = Field(min_length=1, description="Dense embedding model name.")
    revision: str = Field(
        min_length=1,
        description="Configured dense model artifact revision.",
    )
    dimension: int = Field(
        ge=1,
        description="Number of values in every dense vector.",
    )
    normalized: bool = Field(
        description="Whether dense vectors are L2 normalized."
    )
    embeddings: list[list[float]] = Field(
        description="Dense vectors aligned with the request text order."
    )
    sparse_model: str = Field(
        min_length=1,
        description="Sparse-vector algorithm identifier.",
    )
    sparse_embeddings: list[SparseEmbeddingVector] = Field(
        description="Sparse vectors aligned with the request text order."
    )

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

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={"examples": [{"detail": "embedding model is not ready"}]},
    )

    detail: str = Field(
        min_length=1,
        description="Safe client-facing error explanation.",
    )


class HealthResponse(BaseModel):
    """Process liveness or model-readiness status."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={"examples": [{"status": "ready"}]},
    )

    status: Literal["alive", "ready", "not_ready"] = Field(
        description="Current state reported by the selected health probe."
    )
