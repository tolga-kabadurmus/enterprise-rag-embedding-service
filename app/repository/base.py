"""Shared repository lifecycle contract."""

from abc import ABC, abstractmethod


class Repository(ABC):
    """Small asynchronous cleanup contract for infrastructure adapters."""

    @abstractmethod
    async def close(self) -> None:
        """Release external resources; repeated calls must be safe."""
