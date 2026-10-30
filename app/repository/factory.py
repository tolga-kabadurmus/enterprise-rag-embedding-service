"""Infrastructure repository factory."""

from app.common.settings.s3 import S3Settings
from app.repository.s3.base import S3Repository
from app.repository.s3.boto3_repository import Boto3S3Repository


class RepositoryFactory:
    """Construct configured provider adapters at composition time."""

    @staticmethod
    def create_s3(settings: S3Settings) -> S3Repository:
        """Create the configured RustFS-compatible S3 repository."""

        if settings.provider != "rustfs":
            raise ValueError(f"unsupported S3 provider: {settings.provider}")
        return Boto3S3Repository(settings)
