"""Sanitizing middleware for unexpected FastAPI failures."""

import asyncio
import logging

from fastapi import Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

LOGGER = logging.getLogger(__name__)


class UnhandledExceptionMiddleware(BaseHTTPMiddleware):
    """Log inference failures and return safe non-500 responses."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        """Handle a request without exposing internal exception details."""

        try:
            return await call_next(request)
        except Exception as error:
            if isinstance(error, (TimeoutError, asyncio.TimeoutError)):
                status_code = status.HTTP_504_GATEWAY_TIMEOUT
                detail = "embedding inference timed out"
            elif isinstance(error, ValueError):
                status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
                detail = "embedding request could not be processed"
            else:
                status_code = status.HTTP_503_SERVICE_UNAVAILABLE
                detail = "embedding service is temporarily unavailable"
            LOGGER.exception(
                "embedding request failed method=%s path=%s status=%d",
                request.method,
                request.url.path,
                status_code,
            )
            return JSONResponse(
                status_code=status_code,
                content={"detail": detail},
            )
