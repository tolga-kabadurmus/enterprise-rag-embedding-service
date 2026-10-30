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
    Local --> Runtime[ONNX + tokenizer runtime]
    Runtime --> API[FastAPI /v1/embeddings]
    API --> Indexing[enterprise-rag-indexing]
    API --> Query[enterprise-rag-query]
```

The runtime is process-local. Multiple Granian workers create one model runtime
per worker, so worker count must account for model memory.
