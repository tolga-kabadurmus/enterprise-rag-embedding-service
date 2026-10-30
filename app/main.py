"""FastAPI application for the embedding-service scaffold."""

from contextlib import asynccontextmanager
import logging
from typing import AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.api.models import EmbeddingRequest, EmbeddingResponse
from app.common.settings import ApiSettings, ModelSettings, S3Settings, StorageProvider
from app.common.utility.logging import configure_logging
from app.middleware.exceptions import UnhandledExceptionMiddleware
from app.repository.embedding.base import EmbeddingRuntime
from app.repository.embedding.onnx_runtime import OnnxEmbeddingRuntime
from app.repository.embedding.scaffold import ScaffoldEmbeddingRuntime
from app.repository.s3.base import StorageRepositoryBase
from app.repository.s3.boto3_repository import S3Repository
from app.repository.s3.factory import StorageRepositoryFactory

LOGGER = logging.getLogger(__name__)


def create_app(
    api_settings: ApiSettings | None = None,
    model_settings: ModelSettings | None = None,
    s3_settings: S3Settings | None = None,
    enable_model_startup: bool = True,
) -> FastAPI:
    """Build the FastAPI application and its lifecycle hooks.

    The application synchronizes model artifacts and loads the ONNX runtime
    during lifespan startup. ``enable_model_startup=False`` is reserved for
    unit tests that exercise transport behavior without RustFS.
    """

    resolved_api_settings = api_settings or ApiSettings()
    resolved_model_settings = model_settings or ModelSettings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        """Create and close one process-local embedding runtime."""

        s3_repository = None
        runtime: EmbeddingRuntime = ScaffoldEmbeddingRuntime()
        application.state.embedding_runtime = runtime
        application.state.model_settings = resolved_model_settings
        try:
            if enable_model_startup:
                resolved_s3_settings = s3_settings or S3Settings()
                s3_repository = create_storage_repository(resolved_s3_settings)
                await s3_repository.download_prefix(
                    resolved_model_settings.bucket,
                    resolved_model_settings.key_prefix,
                    resolved_model_settings.local_directory,
                )
                runtime = await OnnxEmbeddingRuntime.load(resolved_model_settings)
                application.state.embedding_runtime = runtime
                application.state.s3_repository = s3_repository
                LOGGER.info(
                    "embedding service started model=%s revision=%s runtime=onnx",
                    resolved_model_settings.model_name,
                    resolved_model_settings.revision,
                )
            else:
                LOGGER.info("embedding service started in transport-test mode")
            yield
        finally:
            await runtime.close()
            if s3_repository is not None:
                await s3_repository.close()
            LOGGER.info("embedding service runtime closed")

    configure_logging(
        level=getattr(logging, resolved_api_settings.log_level),
    )
    application = FastAPI(
        title="Enterprise RAG Embedding Service",
        version="0.1.0-scaffold",
        lifespan=lifespan,
    )
    application.add_middleware(UnhandledExceptionMiddleware)

    @application.get("/health/live")
    async def liveness() -> dict[str, str]:
        """Report that the HTTP process is alive."""

        return {"status": "alive"}

    @application.get("/health/ready")
    async def readiness(request: Request) -> JSONResponse:
        """Report readiness only after a usable runtime is available."""

        runtime: EmbeddingRuntime | None = getattr(
            request.app.state,
            "embedding_runtime",
            None,
        )
        if runtime is None or not runtime.is_ready:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "not_ready"},
            )
        return JSONResponse(content={"status": "ready"})

    @application.post("/v1/embeddings")
    async def embeddings(
        http_request: Request,
        request: EmbeddingRequest,
    ) -> JSONResponse:
        """Generate a bounded batch of document or query embeddings."""

        if len(request.texts) > resolved_api_settings.max_batch_size:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={"detail": "embedding batch is too large"},
            )
        if sum(len(text) for text in request.texts) > resolved_api_settings.max_text_characters:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={"detail": "embedding input is too large"},
            )
        runtime: EmbeddingRuntime | None = getattr(
            http_request.app.state,
            "embedding_runtime",
            None,
        )
        if runtime is None or not runtime.is_ready:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"detail": "embedding model is not ready"},
            )
        vectors = await runtime.embed(request.texts, request.input_type)
        response = EmbeddingResponse(
            model=resolved_model_settings.model_name,
            revision=resolved_model_settings.revision,
            dimension=resolved_model_settings.dimension,
            normalized=True,
            embeddings=vectors,
        )
        return JSONResponse(content=response.model_dump())

    return application


def create_storage_repository(settings: S3Settings) -> StorageRepositoryBase:
    """Register storage providers and construct the configured implementation."""

    factory = StorageRepositoryFactory()
    factory.register(StorageProvider.RUSTFS, S3Repository)
    return factory.create(settings.provider, settings)


app = create_app()
