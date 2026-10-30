"""Explicitly unavailable embedding runtime used by the scaffold."""

from collections.abc import Sequence

from app.repository.embedding.base import EmbeddingRuntime


class ScaffoldEmbeddingRuntime(EmbeddingRuntime):
    """Represent the service before a concrete model runtime is added.

    This class deliberately never fabricates vectors. It allows the API
    lifecycle, health endpoints, and dependency boundaries to be exercised
    before model inference is migrated from the indexing component.
    """

    @property
    def is_ready(self) -> bool:
        """Return ``False`` until a concrete runtime is implemented."""

        return False

    async def embed(
        self,
        texts: Sequence[str],
        input_type: str,
    ) -> list[list[float]]:
        """Reject inference rather than returning fabricated vectors."""

        raise RuntimeError("embedding model runtime is not implemented")

    async def close(self) -> None:
        """Provide an idempotent shutdown hook for the future runtime."""

        return None
