"""Corpus ingestion: server-side paths and browser uploads."""

from __future__ import annotations

from fastapi import APIRouter, File, Query, UploadFile

from app.deps import CurrentUser, PipelineDep, SettingsDep
from app.errors import PayloadTooLarge
from app.models import IngestPathsRequest, IngestResponse

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("/paths", response_model=IngestResponse, summary="Index files already on the server")
def ingest_paths(
    body: IngestPathsRequest, pipeline: PipelineDep, user: CurrentUser
) -> IngestResponse:
    return pipeline.ingest_paths(body.paths, recursive=body.recursive, replace=body.replace)


@router.post("/upload", response_model=IngestResponse, summary="Upload and index documents")
async def upload(
    pipeline: PipelineDep,
    settings: SettingsDep,
    user: CurrentUser,
    files: list[UploadFile] = File(...),
    replace: bool = Query(False, description="Clear the index before adding these files."),
) -> IngestResponse:
    limit = settings.max_upload_mb * 1024 * 1024
    payloads: list[tuple[str, bytes]] = []
    total = 0
    for item in files:
        data = await item.read()
        total += len(data)
        if len(data) > limit or total > limit:
            raise PayloadTooLarge(
                f"Upload exceeds the {settings.max_upload_mb} MB limit.",
                detail=f"{item.filename} pushed the request to {total} bytes.",
            )
        payloads.append((item.filename or "upload.bin", data))
    return pipeline.ingest_uploads(payloads, replace=replace)


@router.get("/documents", summary="List indexed documents")
def documents(pipeline: PipelineDep, user: CurrentUser) -> dict[str, object]:
    return {"documents": pipeline.index.document_summaries(), "chunks": len(pipeline.index)}


@router.delete("/documents/{doc_id}", summary="Remove one document from the index")
def delete_document(doc_id: str, pipeline: PipelineDep, user: CurrentUser) -> dict[str, object]:
    removed = pipeline.index.delete_document(doc_id)
    if removed:
        pipeline.index.save()
    return {"doc_id": doc_id, "chunks_removed": removed}


@router.delete("/documents", summary="Empty the index")
def clear_index(pipeline: PipelineDep, user: CurrentUser) -> dict[str, object]:
    pipeline.index.clear()
    pipeline.index.save()
    return {"status": "cleared", "chunks": 0}
