"""Structured logging, request correlation and Prometheus metrics.

Kept in one module so that every other package can emit telemetry with a single
import and no circular dependencies.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")

_STD_ATTRS = set(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
) | {"asctime", "message", "taskName"}


class JsonFormatter(logging.Formatter):
    """One JSON object per line - the format log drains expect."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": trace_id_var.get(),
        }
        for key, value in record.__dict__.items():
            if key not in _STD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<7} " \
               f"[{trace_id_var.get()}] {record.name}: {record.getMessage()}"
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in _STD_ATTRS and not k.startswith("_")
        }
        if extras:
            base += " " + " ".join(f"{k}={v}" for k, v in extras.items())
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_logs else TextFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, root.level))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# --------------------------------------------------------------------------- #
# Metrics - real Prometheus collectors when the client is installed, otherwise
# no-op shims so the rest of the codebase never has to branch.
# --------------------------------------------------------------------------- #
class _NoopMetric:
    def labels(self, *_args: Any, **_kwargs: Any) -> _NoopMetric:
        return self

    def inc(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def observe(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def set(self, *_args: Any, **_kwargs: Any) -> None:
        return None


try:  # pragma: no cover - exercised implicitly by the metrics endpoint test
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
    REGISTRY = CollectorRegistry()

    REQUESTS = Counter(
        "rag_requests_total", "HTTP requests", ["route", "method", "status"], registry=REGISTRY
    )
    REQUEST_LATENCY = Histogram(
        "rag_request_latency_seconds", "HTTP request latency", ["route"], registry=REGISTRY
    )
    STAGE_LATENCY = Histogram(
        "rag_stage_latency_seconds",
        "Pipeline stage latency",
        ["stage"],
        registry=REGISTRY,
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    )
    QUESTIONS = Counter(
        "rag_questions_total", "Questions answered", ["outcome"], registry=REGISTRY
    )
    ABSTENTIONS = Counter(
        "rag_abstentions_total", "Abstentions", ["reason"], registry=REGISTRY
    )
    GROUNDEDNESS = Histogram(
        "rag_groundedness_score",
        "Fraction of answer sentences supported by cited context",
        registry=REGISTRY,
        buckets=(0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0),
    )
    LLM_TOKENS = Counter(
        "rag_llm_tokens_total", "LLM tokens", ["kind", "model"], registry=REGISTRY
    )
    LLM_COST = Counter(
        "rag_llm_cost_usd_total", "Estimated LLM spend", ["model"], registry=REGISTRY
    )
    LLM_ERRORS = Counter(
        "rag_llm_errors_total", "LLM call failures", ["kind"], registry=REGISTRY
    )
    INDEX_SIZE = Gauge("rag_index_chunks", "Chunks in the index", registry=REGISTRY)
    INDEX_DOCS = Gauge("rag_index_documents", "Documents in the index", registry=REGISTRY)
    RETRIEVED_TOP_SCORE = Histogram(
        "rag_top_score",
        "Score of the best retrieved chunk",
        registry=REGISTRY,
        buckets=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0),
    )
except Exception:  # pragma: no cover - prometheus_client not installed
    PROMETHEUS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    REGISTRY = None

    def generate_latest(_registry: Any = None) -> bytes:  # type: ignore[misc]
        return b"# prometheus_client is not installed\n"

    REQUESTS = REQUEST_LATENCY = STAGE_LATENCY = _NoopMetric()
    QUESTIONS = ABSTENTIONS = GROUNDEDNESS = _NoopMetric()
    LLM_TOKENS = LLM_COST = LLM_ERRORS = _NoopMetric()
    INDEX_SIZE = INDEX_DOCS = RETRIEVED_TOP_SCORE = _NoopMetric()


def render_metrics() -> bytes:
    return generate_latest(REGISTRY) if PROMETHEUS_AVAILABLE else generate_latest()


@contextmanager
def timed(bucket: dict[str, float], stage: str) -> Iterator[None]:
    """Record wall-clock ms for a pipeline stage into ``bucket`` and Prometheus."""
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        bucket[stage] = round(elapsed * 1000, 3)
        STAGE_LATENCY.labels(stage=stage).observe(elapsed)
