"""Question answering and raw retrieval."""

from __future__ import annotations

import time

from fastapi import APIRouter

from app.deps import CurrentUser, PipelineDep
from app.models import (
    AnswerResult,
    ChatRequest,
    SearchRequest,
    SearchResponse,
    SessionResponse,
)

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=AnswerResult, summary="Ask a question about the corpus")
def chat(body: ChatRequest, pipeline: PipelineDep, user: CurrentUser) -> AnswerResult:
    return pipeline.answer(
        question=body.question,
        session_id=body.session_id,
        top_k=body.top_k,
        mode=body.mode,
        rerank=body.rerank,
        strict=body.strict,
        include_contexts=body.include_contexts,
    )


@router.post("/search", response_model=SearchResponse, summary="Retrieve without generating")
def search(body: SearchRequest, pipeline: PipelineDep, user: CurrentUser) -> SearchResponse:
    started = time.perf_counter()
    results = pipeline.index.search(
        body.query,
        body.top_k,
        body.mode or pipeline.settings.retrieval_mode,
        pipeline.settings.rerank_enabled if body.rerank is None else body.rerank,
    )
    return SearchResponse(
        query=body.query,
        results=results,
        elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
    )


@router.get(
    "/sessions/{session_id}",
    response_model=SessionResponse,
    summary="Replay a conversation",
)
def session(session_id: str, pipeline: PipelineDep, user: CurrentUser) -> SessionResponse:
    turns = pipeline.sessions.history(session_id, pipeline.settings.history_turns * 4)
    return SessionResponse(session_id=session_id, turns=turns)


@router.delete("/sessions/{session_id}", summary="Forget a conversation")
def reset_session(session_id: str, pipeline: PipelineDep, user: CurrentUser) -> dict[str, object]:
    return {"session_id": session_id, "cleared": pipeline.sessions.reset(session_id)}
