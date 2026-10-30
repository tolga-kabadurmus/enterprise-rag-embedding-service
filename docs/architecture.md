# Architecture and lifecycle

`app/main.py` composes the FastAPI application, settings, S3 repository, and
embedding runtime. The lifespan handler downloads every object under the
configured model prefix, loads the tokenizer and ONNX session, and stores both
in `app.state`.

The S3 repository is a local copy of the indexing repository pattern for now.
It uses boto3 through bounded `asyncio.to_thread()` calls, paginates object
listing, writes downloads through partial files, and rejects unsafe paths.
Cross-repository extraction is intentionally deferred.

```mermaid
flowchart LR
    RustFS[(RustFS models bucket)] -->|paginated prefix download| S3[Embedding S3 repository]
    S3 -->|all model files| Local[/models/embedding/intfloat/multilingual-e5-small]
    Local --> Runtime[ONNX + shared tokenizer runtime]
    Runtime --> Dense[Normalized E5 dense vectors]
    Runtime --> Sparse[Token-frequency sparse vectors]
    Dense --> API[FastAPI /v1/embeddings]
    Sparse --> API
    API -->|dense + sparse batches| Indexing[enterprise-rag-indexing]
    API -->|dense + sparse batches| Query[enterprise-rag-query]
```

Dense preparation applies the E5 `passage:` or `query:` prefix. Sparse
preparation deliberately encodes raw text without those prefixes, counts
non-padding token IDs, and returns sorted indices with positive frequencies.
The API validates that dense and sparse batches remain one-to-one.

The runtime is process-local. Multiple Granian workers create one model runtime
per worker, so worker count must account for model memory.

`UnhandledExceptionMiddleware` logs inference failures without returning their
details. Invalid inference input returns `422`, inference timeouts return `504`,
and unexpected runtime failures return a sanitized `503` instead of HTTP 500.
