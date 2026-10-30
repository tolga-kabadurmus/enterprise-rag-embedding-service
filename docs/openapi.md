# OpenAPI and interactive API documentation

The embedding service publishes an OpenAPI 3 contract generated directly from
its FastAPI routes and Pydantic models. The contract documents operation IDs,
tags, request limits, response models, error statuses, field descriptions, and
example payloads.

With the local Compose service running on its default host port:

| Resource | URL |
| --- | --- |
| Swagger UI | <http://localhost:8081/docs> |
| ReDoc | <http://localhost:8081/redoc> |
| OpenAPI JSON | <http://localhost:8081/openapi.json> |

The documented operations are:

| Method and path | Purpose |
| --- | --- |
| `GET /health/live` | Verify that the HTTP process is alive. |
| `GET /health/ready` | Verify that the embedding runtime is ready. |
| `POST /v1/embeddings` | Create aligned dense and sparse vectors. |

Swagger UI can execute requests against a running local service. Callers must
select `document` for chunks being indexed and `query` for user questions.
Authentication and authorization are outside the project scope, so the
OpenAPI contract intentionally defines no security scheme.

The checked contract tests inspect `/openapi.json` to prevent accidental loss
of endpoint descriptions, response schemas, or stable operation IDs.
