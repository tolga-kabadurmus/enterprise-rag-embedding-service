"""Embedding runtime interface."""

from abc import ABC, abstractmethod
from collections.abc import Sequence


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
    async def close(self) -> None:
        """Release model resources during application shutdown."""
