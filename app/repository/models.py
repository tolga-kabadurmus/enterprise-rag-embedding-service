"""Pydantic records returned by the S3 repository."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class S3ObjectMetadata(BaseModel):
    """Metadata for one object discovered in RustFS."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bucket: str = Field(min_length=1)
    key: str = Field(min_length=1)
    size: int = Field(ge=0)
    etag: str | None = None
    content_type: str | None = None
    last_modified: datetime | None = None
