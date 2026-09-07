"""Integration tests for the full ingest -> retrieve -> answer -> verify path."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.models import RetrievalMode
from app.pipeline import RagPipeline

HANDBOOK = """# Refund policy

Customers may request a refund within 30 days of purchase.
Refunds are issued to the original payment method.

## Exceptions

Digital downloads are non-refundable once accessed.
Shipping fees are never refunded.

## Processing time

Approved refunds settle within 5 to 10 business days.
"""

SECURITY = """# Security policy

All production access requires hardware multi-factor authentication.
Passwords must be at least 14 characters long.

## Incident response

A Severity 1 incident pages the on-call engineer within 5 minutes.
Post-incident reviews are published within 3 business days.
"""


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir(exist_ok=True)  # the settings fixture's ensure_dirs() may have made it
    (root / "handbook.md").write_text(HANDBOOK, encoding="utf-8")
    (root / "security.md").write_text(SECURITY, encoding="utf-8")
    return root


@pytest.fixture
def pipeline(settings: Settings, corpus: Path) -> RagPipeline:
    settings.persist_index = False
    pipe = RagPipeline(settings)
    pipe.index.clear()
    pipe.ingest_paths([corpus], recursive=True)
    return pipe


def test_ingest_indexes_every_document(pipeline: RagPipeline) -> None:
    stats = pipeline.index.stats()
    assert stats["documents"] == 2
    assert stats["chunks"] >= 2
    assert stats["vectors"] == stats["chunks"]
    assert stats["bm25_documents"] == stats["chunks"]


def test_ingest_is_idempotent(pipeline: RagPipeline, corpus: Path) -> None:
    before = len(pipeline.index)
    pipeline.ingest_paths([corpus], recursive=True)
    assert len(pipeline.index) == before, "re-ingesting the same files must replace, not duplicate"


def test_answer_cites_the_right_document(pipeline: RagPipeline) -> None:
    result = pipeline.answer("How many days do customers have to request a refund?")
    assert not result.groundedness.abstained
    assert result.citations, "an answerable question must produce at least one citation"
    assert any("refund" in c.title.lower() or "handbook" in c.source.lower() for c in result.citations)
    assert "30" in result.answer


def test_every_citation_marker_resolves(pipeline: RagPipeline) -> None:
    result = pipeline.answer("What happens during a Severity 1 incident?")
    markers = {c.marker for c in result.citations}
    assert markers == set(range(1, len(result.citations) + 1)), "markers must be dense and 1-based"
    assert not result.groundedness.invalid_citations


def test_unanswerable_question_abstains(pipeline: RagPipeline) -> None:
    result = pipeline.answer("What is the company's policy on cryptocurrency payments?")
    assert result.groundedness.abstained or not result.citations, (
        f"expected an abstention, got: {result.answer!r}"
    )


def test_empty_index_abstains(settings: Settings) -> None:
    settings.persist_index = False
    pipe = RagPipeline(settings)
    pipe.index.clear()
    result = pipe.answer("anything at all")
    assert result.groundedness.abstained
    assert result.answer == settings.abstain_message


def test_conversation_memory_condenses_follow_ups(pipeline: RagPipeline) -> None:
    session = "s1"
    pipeline.answer("How many days do customers have to request a refund?", session_id=session)
    follow_up = pipeline.answer("Are there any exceptions?", session_id=session)

    history = pipeline.sessions.history(session, 10)
    assert len(history) == 4, "each turn stores one user and one assistant message"
    # The offline condenser prepends the prior question's terms, which is what
    # lets a pronoun-only follow-up retrieve anything at all.
    assert "refund" in follow_up.standalone_question.lower()


def test_retrieval_modes_all_return_results(pipeline: RagPipeline) -> None:
    for mode in (RetrievalMode.DENSE, RetrievalMode.LEXICAL, RetrievalMode.HYBRID):
        hits = pipeline.index.search("refund processing time", 3, mode, rerank=False)
        assert hits, f"{mode} returned nothing"
        assert [h.rank for h in hits] == list(range(1, len(hits) + 1))


def test_rerank_does_not_change_result_count(pipeline: RagPipeline) -> None:
    plain = pipeline.index.search("password length requirement", 3, RetrievalMode.HYBRID, False)
    reranked = pipeline.index.search("password length requirement", 3, RetrievalMode.HYBRID, True)
    assert len(plain) == len(reranked)
    assert all(h.rerank_score is not None for h in reranked)


def test_delete_document_removes_it_from_every_index(pipeline: RagPipeline) -> None:
    doc_id = next(iter(pipeline.index.documents))
    removed = pipeline.index.delete_document(doc_id)
    assert removed > 0
    assert doc_id not in pipeline.index.documents
    assert len(pipeline.index.store) == len(pipeline.index)
    assert len(pipeline.index.bm25) == len(pipeline.index)


def test_index_round_trips_through_disk(settings: Settings, corpus: Path) -> None:
    settings.persist_index = True
    first = RagPipeline(settings)
    first.index.clear()
    first.ingest_paths([corpus], recursive=True)
    expected = len(first.index)
    first.index.save()

    second = RagPipeline(settings)
    assert len(second.index) == expected
    hits = second.index.search("refund", 3, RetrievalMode.HYBRID, rerank=False)
    assert hits, "a restored index must still be searchable"


def test_answer_reports_latency_and_usage(pipeline: RagPipeline) -> None:
    result = pipeline.answer("How long do approved refunds take to settle?")
    assert {"retrieve", "generate", "verify"} <= set(result.latency_ms)
    assert all(v >= 0 for v in result.latency_ms.values())
    assert result.usage.total_tokens >= 0
    assert result.trace_id
