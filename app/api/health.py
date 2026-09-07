"""Liveness, readiness and index introspection."""

from __future__ import annotations

from fastapi import APIRouter, Response

from app.deps import PipelineDep, SettingsDep
from app.models import HealthResponse
from app.observability import CONTENT_TYPE_LATEST, render_metrics

router = APIRouter(tags=["ops"])


@router.get("/health", response_model=HealthResponse, summary="Service and index health")
def health(pipeline: PipelineDep) -> HealthResponse:
    return HealthResponse(**pipeline.health())


@router.get("/live", summary="Liveness probe")
def live() -> dict[str, str]:
    """Deliberately does not touch the pipeline: a wedged index should not
    cause an orchestrator to kill an otherwise healthy process."""
    return {"status": "alive"}


@router.get("/ready", summary="Readiness probe")
def ready(pipeline: PipelineDep) -> dict[str, object]:
    return {"status": "ready" if len(pipeline.index) else "empty", "chunks": len(pipeline.index)}


@router.get("/metrics", summary="Prometheus metrics", include_in_schema=False)
def metrics(settings: SettingsDep) -> Response:
    if not settings.metrics_enabled:
        return Response(status_code=404)
    return Response(content=render_metrics(), media_type=CONTENT_TYPE_LATEST)
