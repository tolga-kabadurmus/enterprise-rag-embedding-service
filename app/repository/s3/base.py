"""Provider-neutral S3 repository contract."""

from abc import abstractmethod
from pathlib import Path

from app.repository.base import Repository
from app.repository.models import S3ObjectMetadata


class StorageRepositoryBase(Repository):
    """Async operations required to synchronize model artifacts."""

    @abstractmethod
    async def list_objects(self, bucket: str, key_prefix: str | None = None) -> list[S3ObjectMetadata]:
        """Return every object from all paginated S3 responses."""

    @abstractmethod
    async def download_object(self, bucket: str, object_key: str, destination: Path) -> Path:
        """Download one object atomically to a local path."""

    @abstractmethod
    async def download_prefix(self, bucket: str, key_prefix: str, destination_directory: Path) -> list[Path]:
        """Download all objects below a prefix with safe relative paths."""
