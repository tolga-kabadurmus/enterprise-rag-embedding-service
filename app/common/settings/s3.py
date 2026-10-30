"""Validated RustFS/S3 settings for model artifact synchronization."""

from typing import Literal

from pydantic import AnyHttpUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class S3Settings(BaseSettings):
    """S3 connection and bounded model-download configuration."""

    model_config = SettingsConfigDict(
        env_prefix="S3_",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    provider: Literal["rustfs"] = "rustfs"
    endpoint_url: AnyHttpUrl
    access_key: str = Field(min_length=1)
    secret_key: SecretStr
    region: str = Field(default="us-east-1", min_length=1)
    addressing_style: Literal["path", "virtual", "auto"] = "path"
    verify_tls: bool = True
    max_concurrency: int = Field(default=4, ge=1, le=32)
