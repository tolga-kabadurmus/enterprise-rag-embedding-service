"""Focused tests for the embedding-service scaffold."""

import unittest

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.models import EmbeddingRequest
from app.common.settings import ApiSettings, ModelSettings
from app.main import create_app
from app.middleware.exceptions import UnhandledExceptionMiddleware


class EmbeddingRequestTests(unittest.TestCase):
    """Validate the future request contract without inference."""

    def test_accepts_document_and_query_input_types(self) -> None:
        self.assertEqual(
            EmbeddingRequest(
                input_type="document",
                texts=["a document chunk"],
            ).input_type,
            "document",
        )
        self.assertEqual(
            EmbeddingRequest(
                input_type="query",
                texts=["a question"],
            ).input_type,
            "query",
        )

    def test_rejects_empty_texts_and_unknown_fields(self) -> None:
        with self.assertRaises(ValidationError):
            EmbeddingRequest(input_type="query", texts=[])
        with self.assertRaises(ValidationError):
            EmbeddingRequest(
                input_type="query",
                texts=["question"],
                unexpected="value",
            )


class SettingsTests(unittest.TestCase):
    """Validate defaults and model-directory safety."""

    def test_defaults_match_shared_model_convention(self) -> None:
        self.assertEqual(
            ModelSettings().model_name,
            "intfloat/multilingual-e5-small",
        )
        self.assertEqual(ModelSettings().bucket, "models")
        self.assertEqual(
            ModelSettings().key_prefix,
            "embeddings/intfloat/multilingual-e5-small",
        )
        self.assertEqual(
            str(ModelSettings().local_directory),
            "/models/embedding/intfloat/multilingual-e5-small",
        )
        self.assertEqual(ApiSettings().port, 8081)


class FastApiScaffoldTests(unittest.TestCase):
    """Verify lifecycle, readiness, middleware, and placeholder behavior."""

    def setUp(self) -> None:
        self.client = TestClient(create_app(enable_model_startup=False))

    def tearDown(self) -> None:
        self.client.close()

    def test_liveness_is_available(self) -> None:
        response = self.client.get("/health/live")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "alive"})

    def test_readiness_is_false_until_model_runtime_exists(self) -> None:
        response = self.client.get("/health/ready")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"status": "not_ready"})

    def test_embeddings_endpoint_reports_unavailable_runtime(self) -> None:
        response = self.client.post(
            "/v1/embeddings",
            json={"input_type": "query", "texts": ["question"]},
        )
        self.assertEqual(response.status_code, 503)
        self.assertIn("not ready", response.json()["detail"])

    def test_validation_errors_are_safe(self) -> None:
        response = self.client.post(
            "/v1/embeddings",
            json={"input_type": "invalid", "texts": ["question"]},
        )
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("/home/", response.text)


class ExceptionMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    """Verify inference failures use sanitized non-500 status codes."""

    async def _request_for(self, error: Exception) -> httpx.Response:
        application = FastAPI()
        application.add_middleware(UnhandledExceptionMiddleware)

        @application.get("/failure")
        async def failure() -> None:
            raise error

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(
                app=application,
                raise_app_exceptions=False,
            ),
            base_url="http://test",
        ) as client:
            return await client.get("/failure")

    async def test_maps_inference_timeout(self) -> None:
        response = await self._request_for(TimeoutError("private timeout"))

        self.assertEqual(response.status_code, 504)
        self.assertNotIn("private timeout", response.text)

    async def test_maps_inference_failure_without_http_500(self) -> None:
        response = await self._request_for(RuntimeError("private runtime detail"))

        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private runtime detail", response.text)


if __name__ == "__main__":
    unittest.main()
