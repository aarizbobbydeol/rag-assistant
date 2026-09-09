"""What the service is allowed to keep about the people using it.

These are defaults, not capabilities: each of these behaviours can be turned on
by an operator who wants it, and none of them turns itself on because some
convenience needed it. They are tested because a privacy default that nothing
asserts is one refactor away from silently flipping.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.config import Settings
from app.pipeline import RagPipeline

RUNBOOK = b"# Runbook\n\nA SEV-1 page is acknowledged within 5 minutes.\n"


def test_uploads_are_not_written_to_disk_by_default(settings: Settings):
    """The text is already in the index; the original file is a liability."""
    pipeline = RagPipeline(settings)
    pipeline.ingest_uploads([("runbook.md", RUNBOOK)])

    leftovers = list(Path(settings.upload_dir).iterdir())
    assert leftovers == [], f"upload survived on disk: {leftovers}"


def test_an_operator_can_still_opt_into_keeping_uploads(settings: Settings):
    pipeline = RagPipeline(settings.model_copy(update={"store_uploads": True}))
    pipeline.ingest_uploads([("runbook.md", RUNBOOK)])

    assert [p.name for p in Path(settings.upload_dir).iterdir()] == ["runbook.md"]


def test_the_question_is_not_logged_by_default(settings: Settings, caplog):
    """Logs are shipped and retained far longer than a request lives."""
    pipeline = RagPipeline(settings)
    pipeline.ingest_uploads([("runbook.md", RUNBOOK)])
    secret = "what is my employee identification number 12345"

    with caplog.at_level(logging.INFO):
        pipeline.answer(secret)

    recorded = " ".join(record.getMessage() + str(record.__dict__) for record in caplog.records)
    assert "12345" not in recorded
    assert "employee identification" not in recorded


def test_conversation_memory_never_touches_disk(settings: Settings):
    """Turns live in a bounded in-memory store, so a restart forgets them."""
    pipeline = RagPipeline(settings)
    pipeline.ingest_uploads([("runbook.md", RUNBOOK)])
    pipeline.answer("How quickly is a SEV-1 acknowledged?", session_id="s1")

    assert pipeline.sessions.history("s1", 10), "the turn should be in memory"

    revived = RagPipeline(settings)
    assert revived.sessions.history("s1", 10) == [], "a restart must forget the conversation"
