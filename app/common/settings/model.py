"""Validated embedding model settings for the future runtime."""

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelSettings(BaseSettings):
    """Location and identity of the embedding model.

    The scaffold does not load or execute this model yet. The settings define
    the boundary that the implementation will use during startup.
    """

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDINGS_",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    model_name: str = Field(
        default="intfloat/multilingual-e5-small",
        min_length=1,
    )
    revision: str = Field(default="main", min_length=1)
    local_directory: Path = Path(
        "/models/embedding/intfloat/multilingual-e5-small"
    )
    bucket: str = Field(default="models", min_length=1)
    key_prefix: str = Field(
        default="embeddings/intfloat/multilingual-e5-small",
        min_length=1,
    )
    onnx_model_file: Path = Path("onnx/model.onnx")
    tokenizer_file: Path = Path("tokenizer.json")
    dimension: int = Field(default=384, ge=1)
    sparse_model_name: str = Field(
        default="multilingual-e5-token-frequency-v1",
        min_length=1,
    )

    @field_validator("bucket", "key_prefix")
    @classmethod
    def normalize_s3_path_part(cls, value: str) -> str:
        """Normalize a bucket or prefix and reject blank values."""

        normalized = value.strip().strip("/")
        if not normalized:
            raise ValueError("value must not be blank")
        return normalized

    @field_validator("local_directory")
    @classmethod
    def require_absolute_directory(cls, value: Path) -> Path:
        """Require a container-absolute model directory."""

        if not value.is_absolute():
            raise ValueError("local_directory must be an absolute path")
        return value

    @field_validator("onnx_model_file", "tokenizer_file")
    @classmethod
    def require_safe_relative_artifact(cls, value: Path) -> Path:
        """Require model artifacts to remain below local_directory."""

        if value.is_absolute() or ".." in value.parts or not value.name:
            raise ValueError("model artifact path must be safely relative")
        return value

    @property
    def onnx_model_path(self) -> Path:
        """Return the absolute ONNX model path."""

        return self.local_directory / self.onnx_model_file

    @property
    def tokenizer_path(self) -> Path:
        """Return the absolute tokenizer JSON path."""

        return self.local_directory / self.tokenizer_file
