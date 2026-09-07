"""Index statistics and the effective runtime configuration."""

from __future__ import annotations

from fastapi import APIRouter

from app.deps import CurrentUser, PipelineDep, SettingsDep

router = APIRouter(prefix="/admin", tags=["ops"])

# Anything whose name matches one of these never leaves the process.
_SECRET_HINTS = ("key", "secret", "password", "token", "dsn")


@router.get("/stats", summary="Index and retrieval statistics")
def stats(pipeline: PipelineDep, user: CurrentUser) -> dict[str, object]:
    return {
        "index": pipeline.index.stats(),
        "sessions": len(pipeline.sessions),
        "documents": pipeline.index.document_summaries(),
    }


@router.get("/config", summary="Effective configuration with secrets redacted")
def config(settings: SettingsDep, user: CurrentUser) -> dict[str, object]:
    payload: dict[str, object] = {}
    for name, value in settings.model_dump().items():
        if any(hint in name for hint in _SECRET_HINTS):
            payload[name] = "***set***" if value else ""
        else:
            payload[name] = str(value) if not isinstance(value, (int, float, bool, list)) else value
    return payload
