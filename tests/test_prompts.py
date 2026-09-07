"""Prompt rendering, context budgeting and message assembly."""

from __future__ import annotations

import re

import pytest

from app.generation.prompts import (
    ABSTENTION_SENTENCE,
    CONTEXT_HEADER,
    NO_CONTEXT,
    QUESTION_PREFIX,
    SYSTEM_PROMPT,
    build_answer_messages,
    build_condense_messages,
    build_rerank_messages,
    format_context,
)
from app.models import Chunk, ScoredChunk, Turn
from app.utils import estimate_tokens

MARKER_RE = re.compile(r"^\[(\d+)\]", re.MULTILINE)


@pytest.fixture
def contexts(scored_factory):
    return [
        scored_factory("c1", "Refunds are issued within 30 days.", 0.9, 1),
        scored_factory("c2", "Digital downloads are non-refundable.", 0.7, 2),
        scored_factory("c3", "Shipping fees are never refunded.", 0.5, 3),
    ]


def test_system_prompt_states_the_rules() -> None:
    assert ABSTENTION_SENTENCE in SYSTEM_PROMPT
    lowered = SYSTEM_PROMPT.lower()
    assert "context" in lowered
    assert "[2]" in SYSTEM_PROMPT  # shows the marker style by example
    assert "never" in lowered


def test_format_context_numbers_from_one_and_keeps_order(contexts) -> None:
    rendered, included = format_context(contexts, max_tokens=4000)

    assert included == contexts
    assert [int(m) for m in MARKER_RE.findall(rendered)] == [1, 2, 3]
    assert rendered.startswith("[1] Doc (chunk 0) :: doc.md\n")
    for context in contexts:
        assert context.chunk.text in rendered


def test_format_context_renders_the_page_when_there_is_one(chunk_factory) -> None:
    scored = ScoredChunk(chunk=chunk_factory("c1", "Paged text.", 0, page=7))
    rendered, _ = format_context([scored], max_tokens=500)
    assert rendered.startswith("[1] Doc (p.7) :: doc.md\n")


def test_format_context_returns_only_what_it_rendered(contexts) -> None:
    """The dropped context must leave no marker behind - citations map positionally."""
    budget = estimate_tokens(
        f"[1] {contexts[0].chunk.locator()} :: {contexts[0].chunk.source}\n"
        f"{contexts[0].chunk.text}"
    ) + 2
    rendered, included = format_context(contexts, max_tokens=budget)

    assert len(included) == 1
    assert included[0] is contexts[0]
    assert [int(m) for m in MARKER_RE.findall(rendered)] == [1]
    assert contexts[1].chunk.text not in rendered
    assert estimate_tokens(rendered) <= budget


def test_format_context_markers_always_index_the_returned_list(contexts) -> None:
    for budget in range(4, 200, 7):
        rendered, included = format_context(contexts, budget)
        markers = [int(m) for m in MARKER_RE.findall(rendered)]
        assert markers == list(range(1, len(included) + 1))


def test_format_context_truncates_rather_than_dropping_the_best_passage(
    scored_factory,
) -> None:
    big = scored_factory("big", " ".join(f"sentence number {i}." for i in range(400)))
    rendered, included = format_context([big], max_tokens=60)

    assert len(included) == 1
    assert estimate_tokens(rendered) <= 60
    assert rendered.startswith("[1] ")
    assert len(rendered) < len(big.chunk.text)


def test_format_context_handles_nothing_to_render(contexts) -> None:
    assert format_context([], 1000) == ("", [])
    assert format_context(contexts, 0) == ("", [])


def test_build_answer_messages_shape(contexts) -> None:
    messages = build_answer_messages("What is the refund window?", contexts, (), 4000)

    assert messages[0].role == "system"
    assert messages[0].content == SYSTEM_PROMPT
    assert messages[-1].role == "user"
    assert CONTEXT_HEADER in messages[-1].content
    assert f"{QUESTION_PREFIX} What is the refund window?" in messages[-1].content
    assert "[1]" in messages[-1].content


def test_build_answer_messages_folds_history_in_as_turns(contexts) -> None:
    history = [
        Turn(role="user", content="Do you refund?"),
        Turn(role="assistant", content="Yes, within 30 days [1]."),
    ]
    messages = build_answer_messages("And digital goods?", contexts, history, 4000)

    assert [m.role for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1].content == "Do you refund?"
    assert messages[2].content == "Yes, within 30 days [1]."


def test_build_answer_messages_marks_an_empty_context(contexts) -> None:
    messages = build_answer_messages("Anything?", [], (), 4000)
    assert NO_CONTEXT in messages[-1].content
    assert "[1]" not in messages[-1].content


def test_build_condense_messages_quotes_the_transcript() -> None:
    history = [Turn(role="user", content="Tell me about refunds.")]
    messages = build_condense_messages("What about digital ones?", history)

    assert [m.role for m in messages] == ["system", "user"]
    assert "Tell me about refunds." in messages[1].content
    assert "What about digital ones?" in messages[1].content

    bare = build_condense_messages("Standalone already?")
    assert "Standalone already?" in bare[1].content


def test_build_rerank_messages_numbers_candidates(contexts) -> None:
    messages = build_rerank_messages("refund window", contexts)

    assert [m.role for m in messages] == ["system", "user"]
    assert "refund window" in messages[1].content
    assert [int(m) for m in MARKER_RE.findall(messages[1].content)] == [1, 2, 3]


def test_build_rerank_messages_truncates_long_candidates() -> None:
    long_text = "word " * 5000
    candidate = ScoredChunk(
        chunk=Chunk(
            chunk_id="c",
            doc_id="d",
            source="s.md",
            title="S",
            text=long_text,
        )
    )
    messages = build_rerank_messages("q", [candidate])
    assert len(messages[1].content) < len(long_text)
