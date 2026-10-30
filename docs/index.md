# Enterprise RAG Embedding Service

This service synchronizes the configured embedding model from RustFS during
FastAPI lifespan startup, loads one ONNX runtime per process, and exposes
`POST /v1/embeddings` for document and query vectors.

The local development model location is:

```text
s3://models/embeddings/intfloat/multilingual-e5-small/
    -> /models/embedding/intfloat/multilingual-e5-small/
```

The service applies `passage: ` to document chunks and `query: ` to user
questions before tokenization. It performs attention-mask-aware mean pooling
and L2 normalization. It does not discover documents, chunk PDFs, persist
Qdrant points, retrieve context, or call an LLM.

The S3 settings are supplied through the root environment file
`dev-envs/enterprise-rag-embedding-service.env`:

```text
S3_ENDPOINT_URL
S3_ACCESS_KEY
S3_SECRET_KEY
S3_REGION
S3_ADDRESSING_STYLE
S3_VERIFY_TLS
S3_MAX_CONCURRENCY
EMBEDDINGS_BUCKET=models
EMBEDDINGS_KEY_PREFIX=embeddings/intfloat/multilingual-e5-small
EMBEDDINGS_LOCAL_DIRECTORY=/models/embedding/intfloat/multilingual-e5-small
```

## Lifecycle

Startup fails if S3 synchronization, tokenizer loading, or ONNX model loading
fails. Readiness is unavailable until all startup work completes. Shutdown
closes the ONNX runtime and S3 client.

Inference failures are sanitized: invalid input returns `422`, runtime
unavailability returns `503`, and inference timeout returns `504`. Internal
exceptions, model inputs, paths, and stack traces are not returned to clients.

## Local checks

From the orchestration repository:

```bash
docker compose --profile dev build enterprise-rag-embedding-service
docker compose --profile dev up -d enterprise-rag-embedding-service
curl http://localhost:8081/health/ready
```

The indexer and query consumers do not yet call this service automatically.
