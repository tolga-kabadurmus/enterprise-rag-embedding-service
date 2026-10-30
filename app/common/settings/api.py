"""Validated HTTP service settings."""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    """Settings for the embedding API process.

    Values are read from variables prefixed with ``EMBEDDING_SERVICE_``.
    """

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_SERVICE_",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    host: str = Field(default="0.0.0.0", min_length=1)
    port: int = Field(default=8081, ge=1, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    max_batch_size: int = Field(default=32, ge=1, le=256)
    max_text_characters: int = Field(default=32_000, ge=1, le=1_000_000)
