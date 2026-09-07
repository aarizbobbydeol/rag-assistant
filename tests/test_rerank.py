"""Reranker behaviour: the heuristic's ordering, the shared output contract and
the fallback paths that keep an optional backend from failing a request."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from app.config import Settings
from app.errors import ConfigurationError, LLMError
from app.generation.llm import ExtractiveLLM
from app.models import Chunk, LLMMessage, LLMResponse, ScoredChunk
from app.retrieval.rerank import (
    CrossEncoderReranker,
    HeuristicReranker,
    LLMReranker,
    NoopReranker,
    Reranker,
    get_reranker,
)

QUERY = "how long do approved refunds take to settle"

# Shares exactly one content word with the query, and is handed the winning
# fusion score - the case a reranker exists to correct.
DISTRACTOR = "Refunds are issued to the original payment method for every eligible order."
ANSWER = "Approved refunds settle within 5 to 10 business days."


def ids(results: Sequence[ScoredChunk]) -> list[str]:
    return [result.chunk.chunk_id for result in results]


class FakeLLM(ExtractiveLLM):
    """A chat client whose reply is scripted; `raises` short-circuits the call."""

    provider = "fake"
    # It subclasses ExtractiveLLM only to reuse the plumbing; it stands in for a
    # real chat model, so it must advertise the generative capability.
    generative = True

    def __init__(self, reply: str = "", raises: Exception | None = None) -> None:
        super().__init__(None)
        self.model = "fake"
        self.available = True
        self.reply = reply
        self.raises = raises
        self.calls: list[list[LLMMessage]] = []

    def complete(
        self,
        messages: Sequence[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        if self.raises is not None:
            raise self.raises
        return LLMResponse(text=self.reply)


# --------------------------------------------------------------------------- #
# The heuristic
# --------------------------------------------------------------------------- #
def test_heuristic_promotes_the_chunk_that_answers_the_query(scored_factory) -> None:
    candidates = [
        scored_factory("distractor", DISTRACTOR, score=1.0, rank=1),
        scored_factory("answer", ANSWER, score=0.2, rank=2),
    ]

    results = HeuristicReranker().rerank(QUERY, candidates, top_n=2)

    assert ids(results) == ["answer", "distractor"]
    assert results[0].rerank_score > results[1].rerank_score


def test_heuristic_rewards_terms_that_sit_together(scored_factory) -> None:
    """Same coverage, different spread: the passage where the terms cluster wins."""
    together = scored_factory(
        "together", "Approved refunds settle in ten days.", score=0.5, rank=1
    )
    apart = scored_factory(
        "apart",
        "Approved returns are logged. " + "Unrelated filler sentence here. " * 6
        + "Refunds settle eventually.",
        score=0.5,
        rank=2,
    )

    results = HeuristicReranker().rerank(QUERY, [apart, together], top_n=2)

    assert ids(results) == ["together", "apart"]


def test_heuristic_position_prior_settles_a_tie(chunk_factory) -> None:
    """Identical text and identical fusion score: the earlier chunk wins."""
    candidates = [
        ScoredChunk(chunk=chunk_factory("late", ANSWER, ordinal=9), score=0.5, rank=1),
        ScoredChunk(chunk=chunk_factory("early", ANSWER, ordinal=0), score=0.5, rank=2),
    ]

    results = HeuristicReranker().rerank(QUERY, candidates, top_n=2)

    assert ids(results) == ["early", "late"]


def test_heuristic_scores_stay_in_the_unit_interval(scored_factory) -> None:
    candidates = [
        scored_factory("a", ANSWER, score=0.9, rank=1),
        scored_factory("b", DISTRACTOR, score=0.1, rank=2),
    ]

    for result in HeuristicReranker().rerank(QUERY, candidates, top_n=2):
        assert 0.0 <= result.rerank_score <= 1.0


def test_heuristic_survives_an_all_stopword_query(scored_factory) -> None:
    """No content words means no lexical evidence - fall back on fusion order."""
    candidates = [
        scored_factory("low", DISTRACTOR, score=0.1, rank=2),
        scored_factory("high", ANSWER, score=0.9, rank=1),
    ]

    results = HeuristicReranker().rerank("what about it?", candidates, top_n=2)

    assert ids(results) == ["high", "low"]


def test_heuristic_is_deterministic(scored_factory) -> None:
    candidates = [
        scored_factory(f"c{i}", f"{ANSWER} {i}", score=1.0 / (i + 1), rank=i + 1)
        for i in range(6)
    ]
    reranker = HeuristicReranker()

    first = reranker.rerank(QUERY, candidates, top_n=4)
    second = reranker.rerank(QUERY, candidates, top_n=4)

    assert ids(first) == ids(second)
    assert [r.rerank_score for r in first] == [r.rerank_score for r in second]


# --------------------------------------------------------------------------- #
# The output contract, which every backend shares
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "reranker",
    [HeuristicReranker(), NoopReranker(), LLMReranker(FakeLLM(reply="nonsense"))],
    ids=["heuristic", "noop", "llm"],
)
def test_input_list_is_not_mutated(reranker: Reranker, scored_factory) -> None:
    candidates = [
        scored_factory("distractor", DISTRACTOR, score=1.0, rank=1),
        scored_factory("answer", ANSWER, score=0.2, rank=2),
    ]
    before = [candidate.model_dump() for candidate in candidates]

    results = reranker.rerank(QUERY, candidates, top_n=2)

    assert [candidate.model_dump() for candidate in candidates] == before
    assert len(candidates) == 2
    # New objects all the way down, so mutating a result cannot reach the index.
    for result in results:
        assert all(result is not candidate for candidate in candidates)
        assert all(result.chunk is not candidate.chunk for candidate in candidates)
    results[0].chunk.metadata["touched"] = True
    assert all(candidate.chunk.metadata == {} for candidate in candidates)


@pytest.mark.parametrize(
    "reranker",
    [HeuristicReranker(), NoopReranker(), LLMReranker(FakeLLM(reply="nonsense"))],
    ids=["heuristic", "noop", "llm"],
)
def test_truncates_to_top_n_and_renumbers_rank(reranker: Reranker, scored_factory) -> None:
    candidates = [
        scored_factory(
            f"c{i}", f"Approved refunds settle after {i} days.", score=1.0 - i / 10, rank=i + 1
        )
        for i in range(5)
    ]

    results = reranker.rerank(QUERY, candidates, top_n=3)

    assert len(results) == 3
    assert [result.rank for result in results] == [1, 2, 3]
    # `score` is overwritten with the rerank score, and the list is sorted by it.
    assert all(result.score == result.rerank_score for result in results)
    scores = [result.rerank_score for result in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize(
    "reranker",
    [HeuristicReranker(), NoopReranker(), LLMReranker(FakeLLM(reply="[]"))],
    ids=["heuristic", "noop", "llm"],
)
def test_empty_candidates_return_empty(reranker: Reranker) -> None:
    assert reranker.rerank(QUERY, [], top_n=5) == []


def test_noop_preserves_the_fusion_order(scored_factory) -> None:
    """The control arm must not reorder, even when the scores say it should."""
    candidates = [
        scored_factory("first", DISTRACTOR, score=0.1, rank=1),
        scored_factory("second", ANSWER, score=0.9, rank=2),
    ]

    results = NoopReranker().rerank(QUERY, candidates, top_n=2)

    assert ids(results) == ["first", "second"]
    assert [result.rerank_score for result in results] == [0.1, 0.9]


# --------------------------------------------------------------------------- #
# LLM reranking
# --------------------------------------------------------------------------- #
def test_llm_reranker_applies_the_returned_scores(scored_factory) -> None:
    candidates = [
        scored_factory("distractor", DISTRACTOR, score=1.0, rank=1),
        scored_factory("answer", ANSWER, score=0.2, rank=2),
    ]
    llm = FakeLLM(
        reply='Here you go:\n```json\n[{"index": 2, "score": 9}, {"index": 1, "score": 1}]\n```'
    )

    results = LLMReranker(llm).rerank(QUERY, candidates, top_n=2)

    assert ids(results) == ["answer", "distractor"]
    assert results[0].rerank_score == pytest.approx(0.9)
    assert results[1].rerank_score == pytest.approx(0.1)
    assert len(llm.calls) == 1


def test_llm_reranker_ranks_unscored_candidates_last(scored_factory) -> None:
    candidates = [
        scored_factory("a", ANSWER, score=0.9, rank=1),
        scored_factory("b", DISTRACTOR, score=0.8, rank=2),
        scored_factory("c", ANSWER, score=0.7, rank=3),
    ]
    # Only the third passage is judged; indices 4 and 0 are out of range.
    llm = FakeLLM(
        reply='[{"index": 3, "score": 5}, {"index": 4, "score": 10}, {"index": 0, "score": 10}]'
    )

    results = LLMReranker(llm).rerank(QUERY, candidates, top_n=3)

    assert ids(results) == ["c", "a", "b"]
    assert results[0].rerank_score == pytest.approx(0.5)
    assert [result.rerank_score for result in results[1:]] == [0.0, 0.0]


@pytest.mark.parametrize(
    "reply",
    [
        "I'm sorry, I can't help with that.",
        "",
        "[not json at all}",
        '[{"passage": 1, "relevance": 9}]',
        '{"index": 1, "score": 9}',
        '["1", "2"]',
        '[{"index": 1, "score": "very high"}]',
    ],
    ids=["refusal", "empty", "broken", "wrong-keys", "not-a-list", "bare-strings", "non-numeric"],
)
def test_llm_reranker_falls_back_on_a_malformed_response(reply: str, scored_factory) -> None:
    candidates = [
        scored_factory("distractor", DISTRACTOR, score=1.0, rank=1),
        scored_factory("answer", ANSWER, score=0.2, rank=2),
    ]
    expected = HeuristicReranker().rerank(QUERY, candidates, top_n=2)

    results = LLMReranker(FakeLLM(reply=reply)).rerank(QUERY, candidates, top_n=2)

    assert ids(results) == ids(expected)
    assert [r.rerank_score for r in results] == [r.rerank_score for r in expected]


@pytest.mark.parametrize(
    "error",
    [LLMError("provider is down"), RuntimeError("a client raised something exotic")],
    ids=["llm-error", "unexpected"],
)
def test_llm_reranker_never_lets_a_failure_escape(error: Exception, scored_factory) -> None:
    candidates = [
        scored_factory("distractor", DISTRACTOR, score=1.0, rank=1),
        scored_factory("answer", ANSWER, score=0.2, rank=2),
    ]

    results = LLMReranker(FakeLLM(raises=error)).rerank(QUERY, candidates, top_n=2)

    assert ids(results) == ["answer", "distractor"]


def test_llm_reranker_uses_the_shared_rerank_prompt(scored_factory) -> None:
    candidates = [scored_factory("answer", ANSWER, score=0.5, rank=1)]
    llm = FakeLLM(reply='[{"index": 1, "score": 8}]')

    LLMReranker(llm).rerank(QUERY, candidates, top_n=1)

    roles = [message.role for message in llm.calls[0]]
    assert roles == ["system", "user"]
    assert QUERY in llm.calls[0][1].content
    assert ANSWER in llm.calls[0][1].content


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def test_get_reranker_auto_is_the_heuristic(settings: Settings) -> None:
    """`auto` must never opt an operator into a model download or a paid call."""
    reranker = get_reranker(settings.model_copy(update={"rerank_backend": "auto"}))

    assert isinstance(reranker, HeuristicReranker)
    assert reranker.name == "heuristic"


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("heuristic", HeuristicReranker),
        ("HEURISTIC", HeuristicReranker),
        ("none", NoopReranker),
        ("noop", NoopReranker),
        ("off", NoopReranker),
    ],
)
def test_get_reranker_resolves_names(
    settings: Settings, backend: str, expected: type[Reranker]
) -> None:
    assert isinstance(
        get_reranker(settings.model_copy(update={"rerank_backend": backend})), expected
    )


def test_get_reranker_rejects_an_unknown_backend(settings: Settings) -> None:
    with pytest.raises(ConfigurationError):
        get_reranker(settings.model_copy(update={"rerank_backend": "magic"}))


def test_get_reranker_falls_back_when_the_cross_encoder_is_unavailable(
    settings: Settings,
) -> None:
    cfg = settings.model_copy(update={"rerank_backend": "cross-encoder"})

    reranker = get_reranker(cfg)

    # sentence-transformers is optional: installed it is used, missing it degrades.
    assert isinstance(reranker, (CrossEncoderReranker, HeuristicReranker))


def test_get_reranker_will_not_wrap_the_extractive_backend(settings: Settings) -> None:
    """`ExtractiveLLM` cannot follow a scoring instruction, so wrapping it is waste."""
    cfg = settings.model_copy(update={"rerank_backend": "llm"})

    assert isinstance(get_reranker(cfg, ExtractiveLLM(cfg)), HeuristicReranker)
    assert isinstance(get_reranker(cfg, None), HeuristicReranker)


def test_get_reranker_builds_an_llm_reranker_for_a_chat_model(settings: Settings) -> None:
    cfg = settings.model_copy(update={"rerank_backend": "llm"})

    reranker = get_reranker(cfg, FakeLLM(reply="[]"))

    assert isinstance(reranker, LLMReranker)
    assert reranker.name == "llm"


def test_chunk_type_is_preserved(scored_factory) -> None:
    """The copy must still be a `Chunk`, not a dict, for the citation layer."""
    results = HeuristicReranker().rerank(
        QUERY, [scored_factory("a", ANSWER, score=0.5, rank=1)], top_n=1
    )

    assert isinstance(results[0], ScoredChunk)
    assert isinstance(results[0].chunk, Chunk)
    assert results[0].chunk.text == ANSWER
