"""FastAPI application factory.

Middleware order matters here: correlation IDs are attached first so that every
log line and every error response produced further down carries the same
``trace_id`` the client sees in the ``X-Trace-Id`` response header.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import admin, auth, chat, health, ingest
from app.config import get_settings
from app.deps import get_pipeline
from app.errors import RagError
from app.observability import (
    REQUEST_LATENCY,
    REQUESTS,
    configure_logging,
    get_logger,
    trace_id_var,
)
from app.utils import new_trace_id

logger = get_logger(__name__)

DESCRIPTION = """
Ask questions about your own PDFs and internal documents and get answers with
inline `[n]` citations back to the exact passage.

* **Hybrid retrieval** - dense vectors fused with BM25 via reciprocal rank fusion
* **Reranking** - heuristic, cross-encoder or LLM
* **Grounding checks** - every answer sentence is scored against its cited chunk
* **Conversation memory** - follow-ups are condensed into standalone questions
* **Evaluation** - `python -m eval.run_eval` and `python -m eval.sweep`
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    logger.info("starting", extra={"version": settings.version, "port": settings.port})
    pipeline = get_pipeline()  # build the index eagerly so the first request is not slow

    # A container starts with no persisted index, and nothing else ingests the
    # corpus, so without this a deployed instance comes up healthy and abstains
    # on every question - the worst kind of failure, because it looks like it is
    # working. Seeding from the configured corpus makes a fresh deploy useful on
    # its first request, and is a no-op once an index exists.
    if not len(pipeline.index) and settings.corpus_dir.is_dir():
        result = pipeline.ingest_paths([settings.corpus_dir], recursive=True)
        logger.info(
            "seeded_corpus",
            extra={
                "corpus_dir": str(settings.corpus_dir),
                "documents": result.total_documents,
                "chunks": len(pipeline.index),
            },
        )

    yield
    logger.info("shutting_down")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)

    app = FastAPI(
        title="RAG Knowledge Assistant",
        description=DESCRIPTION,
        version=settings.version,
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Trace-Id"],
    )

    @app.middleware("http")
    async def correlate_and_measure(request: Request, call_next: Callable):
        trace_id = request.headers.get("X-Trace-Id") or new_trace_id()
        token = trace_id_var.set(trace_id)
        started = time.perf_counter()
        route = request.url.path
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            REQUESTS.labels(route=route, method=request.method, status="500").inc()
            logger.exception("unhandled_request_error", extra={"route": route})
            raise
        finally:
            elapsed = time.perf_counter() - started
            REQUEST_LATENCY.labels(route=route).observe(elapsed)
            trace_id_var.reset(token)

        REQUESTS.labels(route=route, method=request.method, status=str(status_code)).inc()
        response.headers["X-Trace-Id"] = trace_id
        response.headers["X-Response-Time-ms"] = f"{elapsed * 1000:.1f}"
        return response

    @app.exception_handler(RagError)
    async def rag_error_handler(request: Request, exc: RagError) -> JSONResponse:
        logger.warning(
            "rag_error",
            extra={"code": exc.code, "status": exc.status_code, "path": request.url.path},
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.code,
                "detail": exc.detail or exc.message,
                "trace_id": trace_id_var.get(),
            },
        )

    api_prefix = "/api"
    app.include_router(health.router)
    app.include_router(health.router, prefix=api_prefix)
    app.include_router(auth.router, prefix=api_prefix)
    app.include_router(ingest.router, prefix=api_prefix)
    app.include_router(chat.router, prefix=api_prefix)
    app.include_router(admin.router, prefix=api_prefix)

    _mount_frontend(app)
    return app


def _mount_frontend(app: FastAPI) -> None:
    """Serve the built React SPA when it exists.

    In development the Vite dev server proxies to this API instead, so a missing
    build directory is normal and must not be an error.
    """
    dist = Path(__file__).resolve().parent.parent / "web" / "dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="web")
        logger.info("frontend_mounted", extra={"path": str(dist)})


app = create_app()
