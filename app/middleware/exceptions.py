"""Sanitizing middleware for unexpected FastAPI failures."""

import logging

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

LOGGER = logging.getLogger(__name__)


class UnhandledExceptionMiddleware(BaseHTTPMiddleware):
    """Log unexpected exceptions and return a safe generic response."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        """Handle a request without exposing internal exception details."""

        try:
            return await call_next(request)
        except Exception:
            LOGGER.exception(
                "unhandled embedding-service request method=%s path=%s",
                request.method,
                request.url.path,
            )
            return JSONResponse(
                status_code=500,
                content={"detail": "internal server error"},
            )
