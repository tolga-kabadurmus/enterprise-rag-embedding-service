"""Contract tests for generated OpenAPI and interactive documentation routes."""

import unittest

from app.main import create_app


class OpenApiContractTests(unittest.TestCase):
    """Keep the public embedding-service contract complete and discoverable."""

    def setUp(self) -> None:
        self.application = create_app(enable_model_startup=False)
        self.schema = self.application.openapi()

    def test_exposes_swagger_redoc_and_openapi_routes(self) -> None:
        """Expose both interactive renderers and the machine-readable schema."""

        paths = {route.path for route in self.application.routes}

        self.assertTrue({"/docs", "/redoc", "/openapi.json"} <= paths)

    def test_documents_service_metadata_and_tags(self) -> None:
        """Describe service purpose and group all public operations."""

        self.assertEqual(
            self.schema["info"]["title"],
            "Enterprise RAG Embedding Service",
        )
        self.assertIn("vectorization", self.schema["info"]["description"])
        self.assertEqual(
            {tag["name"] for tag in self.schema["tags"]},
            {"health", "embeddings"},
        )
        self.assertNotIn("securitySchemes", self.schema["components"])

    def test_documents_every_operation_and_error_response(self) -> None:
        """Publish stable IDs, summaries, schemas, and operational failures."""

        paths = self.schema["paths"]
        self.assertEqual(
            paths["/health/live"]["get"]["operationId"],
            "get_liveness",
        )
        self.assertEqual(
            paths["/health/ready"]["get"]["operationId"],
            "get_readiness",
        )
        operation = paths["/v1/embeddings"]["post"]
        self.assertEqual(operation["operationId"], "create_embeddings_v1")
        self.assertTrue(operation["summary"])
        self.assertTrue(operation["description"])
        self.assertEqual(
            set(operation["responses"]),
            {"200", "413", "422", "503", "504"},
        )

    def test_documents_payload_fields_and_examples(self) -> None:
        """Keep schema fields understandable without reading implementation code."""

        schemas = self.schema["components"]["schemas"]
        request_schema = schemas["EmbeddingRequest"]
        response_schema = schemas["EmbeddingResponse"]

        self.assertTrue(request_schema["examples"])
        self.assertTrue(request_schema["properties"]["input_type"]["description"])
        self.assertTrue(response_schema["examples"])
        self.assertTrue(response_schema["properties"]["embeddings"]["description"])


if __name__ == "__main__":
    unittest.main()
