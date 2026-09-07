"""Shared pytest fixtures.

Puts the repository root on ``sys.path`` so tests can ``import app`` without an
editable install, and points every settings-dependent test at a scratch
directory so runs never touch the developer's real index.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.models import Chunk, Document, ScoredChunk  # noqa: E402


@pytest.fixture(autouse=True)
def isolate_settings_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's ``.env`` and shell out of the test run.

    ``Settings`` reads ``.env`` from the working directory, so a real API key or
    a pinned backend on one machine silently changes what the tests assert -
    and CI, which has no ``.env``, then disagrees with the local run. Tests that
    want a non-default setting must say so explicitly.
    """
    for key in [name for name in os.environ if name.startswith("RAG_")]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", None)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Default settings rerouted to a temp directory."""
    cfg = Settings(
        data_dir=tmp_path,
        index_dir=tmp_path / "index",
        corpus_dir=tmp_path / "corpus",
        upload_dir=tmp_path / "uploads",
        auth_db_path=tmp_path / "users.sqlite3",
        auth_secret="test-secret-key",
        llm_backend="extractive",
        embedding_backend="hashing",
        vector_store="numpy",
        rerank_backend="heuristic",
    )
    cfg.ensure_dirs()
    return cfg


@pytest.fixture
def sample_document() -> Document:
    text = (
        "# Refund policy\n\n"
        "Customers may request a refund within 30 days of purchase. "
        "Refunds are issued to the original payment method.\n\n"
        "## Exceptions\n\n"
        "Digital downloads are non-refundable once accessed. "
        "Shipping fees are never refunded.\n\n"
        "## Processing time\n\n"
        "Approved refunds settle within 5 to 10 business days."
    )
    return Document(
        doc_id="doc-refund",
        source="policies/refunds.md",
        title="Refund policy",
        text=text,
        metadata={},
    )


def make_chunk(chunk_id: str, text: str, ordinal: int = 0, page: int | None = None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        source="doc.md",
        title="Doc",
        text=text,
        ordinal=ordinal,
        start_char=0,
        end_char=len(text),
        page=page,
        token_count=max(1, len(text.split())),
    )


def make_scored(chunk_id: str, text: str, score: float = 1.0, rank: int = 1) -> ScoredChunk:
    return ScoredChunk(chunk=make_chunk(chunk_id, text), score=score, rank=rank)


@pytest.fixture
def chunk_factory():
    return make_chunk


@pytest.fixture
def scored_factory():
    return make_scored
