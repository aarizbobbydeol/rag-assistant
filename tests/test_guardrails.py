"""Tests for the hallucination guard: per-sentence support and the answer verdict."""

from __future__ import annotations

import pytest

from app.generation.guardrails import (
    check_groundedness,
    query_term_coverage,
    sentence_support,
    should_abstain,
)
from app.models import Citation
from app.retrieval.embeddings import HashingEmbedder

REFUNDS = (
    "Customers may request a refund within 30 days of purchase. "
    "Digital downloads are non-refundable once accessed. "
    "Shipping fees are never refunded."
)
PROCESSING = "Approved refunds settle within 5 to 10 business days."
GIFT_CARDS = "Gift cards cannot be exchanged for cash."

# Shares no vocabulary with any context - the unambiguous fabrication.
FABRICATED = "The company was founded in Zurich by two former bankers"
# Overlaps a little ("refunds", "days") but asserts something new: the case the
# guard has to get right, not just the easy one.
HALF_FABRICATED = "Refunds are paid in Bitcoin after ninety business days"


@pytest.fixture
def contexts(scored_factory):
    return [
        scored_factory("c1", REFUNDS, score=0.9, rank=1),
        scored_factory("c2", PROCESSING, score=0.7, rank=2),
        scored_factory("c3", GIFT_CARDS, score=0.5, rank=3),
    ]


@pytest.fixture
def embedder():
    return HashingEmbedder(dim=1024)


def cite(marker: int, scored) -> Citation:
    """A Citation as `resolve_citations` would build it."""
    chunk = scored.chunk
    return Citation(
        marker=marker,
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        source=chunk.source,
        title=chunk.title,
        page=chunk.page,
        quote=chunk.text[:80],
        score=scored.score,
    )


def strict(settings):
    return settings.model_copy(update={"strict_grounding": True})


# --------------------------------------------------------------------------- #
# sentence_support
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "sentence,chunk_texts",
    [
        ("", [REFUNDS]),
        ("   ", [REFUNDS]),
        ("Customers may request a refund.", []),
        ("", []),
        ("Here is what I found:", [REFUNDS]),  # scaffolding: no content words
        ("Note the following:", [REFUNDS]),
    ],
)
def test_sentence_support_returns_zero_when_there_is_nothing_to_ground(
    sentence, chunk_texts
):
    assert sentence_support(sentence, chunk_texts) == 0.0


def test_verbatim_sentence_is_fully_supported():
    sentence = "Customers may request a refund within 30 days of purchase."
    assert sentence_support(sentence, [REFUNDS]) == pytest.approx(1.0)


def test_fabricated_sentence_scores_near_zero(settings):
    score = sentence_support(FABRICATED, [REFUNDS, PROCESSING, GIFT_CARDS])
    assert score < settings.groundedness_threshold
    assert score == pytest.approx(0.0)


def test_partly_overlapping_fabrication_stays_below_the_threshold(settings):
    score = sentence_support(HALF_FABRICATED, [REFUNDS, PROCESSING, GIFT_CARDS])
    assert 0.0 < score < settings.groundedness_threshold


def test_support_is_bounded_and_deterministic():
    for sentence in (REFUNDS, FABRICATED, HALF_FABRICATED, PROCESSING):
        first = sentence_support(sentence, [REFUNDS, PROCESSING])
        assert 0.0 <= first <= 1.0
        assert sentence_support(sentence, [REFUNDS, PROCESSING]) == first


def test_markers_do_not_affect_the_score():
    plain = "Shipping fees are never refunded."
    assert sentence_support(f"{plain} [1, 2]", [REFUNDS]) == sentence_support(
        plain, [REFUNDS]
    )


def test_support_never_pools_vocabulary_across_chunks():
    """A claim stitched from two passages is not the same as one passage saying it."""
    stitched = "Gift cards settle within 5 to 10 business days."
    assert sentence_support(stitched, [PROCESSING, GIFT_CARDS]) < 0.9


def test_word_order_matters_less_than_word_presence():
    """Bigrams are the minority term, so a reordering still scores as related."""
    shuffled = "Within 30 days of purchase a refund may be requested by customers."
    score = sentence_support(shuffled, [REFUNDS])
    assert 0.5 < score < 1.0


# --------------------------------------------------------------------------- #
# sentence_support with an embedder
# --------------------------------------------------------------------------- #
def test_embedder_moves_the_score_toward_cosine_but_lexical_dominates(embedder):
    sentence = "Customers may request a refund within 30 days of purchase."
    lexical = sentence_support(sentence, [REFUNDS])
    blended = sentence_support(sentence, [REFUNDS], embedder=embedder)
    cosine = float(embedder.embed_query(sentence) @ embedder.embed_documents([REFUNDS])[0])

    assert cosine < blended < lexical
    # The blend sits nearer the lexical evidence than the embedding evidence,
    # which is the whole point of weighting it higher.
    assert abs(blended - lexical) < abs(blended - cosine)


def test_embedder_can_only_nudge_a_fabrication(embedder, settings):
    blended = sentence_support(FABRICATED, [REFUNDS, PROCESSING], embedder=embedder)
    assert blended < settings.groundedness_threshold


def test_embedder_does_not_flip_the_verdict_on_clear_cut_cases(settings, embedder):
    verbatim = "Shipping fees are never refunded."
    threshold = settings.groundedness_threshold
    for sentence, expected in ((verbatim, True), (FABRICATED, False), (HALF_FABRICATED, False)):
        chunks = [REFUNDS, PROCESSING, GIFT_CARDS]
        assert (sentence_support(sentence, chunks) >= threshold) is expected
        assert (sentence_support(sentence, chunks, embedder) >= threshold) is expected


def test_broken_embedder_degrades_to_lexical_evidence():
    class Exploding(HashingEmbedder):
        def embed_documents(self, texts):  # type: ignore[override]
            raise RuntimeError("backend down")

    sentence = "Shipping fees are never refunded."
    assert sentence_support(sentence, [REFUNDS], Exploding(dim=64)) == sentence_support(
        sentence, [REFUNDS]
    )


# --------------------------------------------------------------------------- #
# check_groundedness - the happy path
# --------------------------------------------------------------------------- #
def test_answer_copied_from_context_is_fully_grounded(contexts, settings):
    answer = (
        "Customers may request a refund within 30 days of purchase [1]. "
        "Approved refunds settle within 5 to 10 business days [2]."
    )
    citations = [cite(1, contexts[0]), cite(2, contexts[1])]

    result = check_groundedness(answer, contexts, citations, [], settings)

    assert result.score == pytest.approx(1.0)
    assert result.total_sentences == 2
    assert result.supported_sentences == 2
    assert result.unsupported == []
    assert result.uncited_sentences == 0
    assert result.invalid_citations == []
    assert result.abstained is False
    assert result.reason is None
    assert [s.cited_markers for s in result.sentences] == [[1], [2]]
    assert all(s.supported for s in result.sentences)


def test_fabricated_sentence_lands_in_unsupported(contexts, settings):
    answer = (
        f"Customers may request a refund within 30 days of purchase [1]. "
        f"{FABRICATED} [1]."
    )
    citations = [cite(1, contexts[0])]

    result = check_groundedness(answer, contexts, citations, [], settings)

    assert result.total_sentences == 2
    assert result.supported_sentences == 1
    assert result.score == pytest.approx(0.5)
    assert len(result.unsupported) == 1
    assert FABRICATED in result.unsupported[0]
    assert result.sentences[1].support_score < settings.groundedness_threshold


def test_scaffolding_sentences_are_skipped_not_counted_as_unsupported(
    contexts, settings
):
    answer = (
        "Here is what I found: "
        "Shipping fees are never refunded [1]."
    )
    result = check_groundedness(answer, contexts, [cite(1, contexts[0])], [], settings)

    assert result.total_sentences == 1
    assert result.score == pytest.approx(1.0)
    assert "Here is what I found:" not in " ".join(result.unsupported)
    assert len(result.sentences) == 1


def test_empty_answer_scores_zero_without_abstaining(contexts, settings):
    result = check_groundedness("", contexts, [], [], settings)

    assert result.total_sentences == 0
    assert result.supported_sentences == 0
    assert result.sentences == []
    assert result.score == 0.0
    assert result.abstained is False


# --------------------------------------------------------------------------- #
# check_groundedness - citations
# --------------------------------------------------------------------------- #
def test_a_sentence_is_judged_against_the_chunk_it_actually_cites(contexts, settings):
    """Citing the wrong passage is a failure even when the claim is in the corpus."""
    claim = "Approved refunds settle within 5 to 10 business days"
    citations = [cite(1, contexts[0]), cite(2, contexts[1])]

    wrong = check_groundedness(f"{claim} [1].", contexts, citations, [], settings)
    right = check_groundedness(f"{claim} [2].", contexts, citations, [], settings)

    assert wrong.supported_sentences == 0
    assert right.supported_sentences == 1
    assert wrong.uncited_sentences == 0  # it did cite something, just the wrong thing


def test_uncited_sentences_are_checked_against_every_context(contexts, settings):
    answer = "Approved refunds settle within 5 to 10 business days."

    result = check_groundedness(answer, contexts, [], [], settings)

    assert result.uncited_sentences == 1
    assert result.total_sentences == 1
    assert result.supported_sentences == 1
    assert result.sentences[0].cited_markers == []


def test_markers_are_resolved_by_chunk_id_not_by_position(contexts, settings):
    """`renumber_answer` densifies markers, so [1] need not be contexts[0]."""
    answer = "Gift cards cannot be exchanged for cash [1]."
    citations = [cite(1, contexts[2])]

    result = check_groundedness(answer, contexts, citations, [], settings)

    assert result.score == pytest.approx(1.0)
    assert result.uncited_sentences == 0


def test_invalid_markers_propagate_and_deduplicate(contexts, settings):
    answer = "Shipping fees are never refunded [7]."

    result = check_groundedness(answer, contexts, [], [7, 9, 7], settings)

    assert result.invalid_citations == [7, 9]


def test_a_sentence_citing_only_invalid_markers_falls_back_to_all_contexts(
    contexts, settings
):
    answer = f"{FABRICATED} [9]."

    result = check_groundedness(answer, contexts, [], [9], settings)

    assert result.uncited_sentences == 1
    assert result.total_sentences == 1
    assert result.supported_sentences == 0
    assert result.sentences[0].cited_markers == [9]


# --------------------------------------------------------------------------- #
# check_groundedness - strict mode and abstention
# --------------------------------------------------------------------------- #
def test_strict_mode_flips_abstained_on_a_weak_answer(contexts, settings):
    answer = (
        f"Shipping fees are never refunded [1]. {FABRICATED} [1]. {HALF_FABRICATED} [1]."
    )
    citations = [cite(1, contexts[0])]

    lenient = check_groundedness(answer, contexts, citations, [], settings)
    guarded = check_groundedness(answer, contexts, citations, [], strict(settings))

    assert lenient.score == guarded.score < settings.min_answer_groundedness
    assert lenient.abstained is False
    assert lenient.reason is None
    assert guarded.abstained is True
    assert guarded.reason is not None and "groundedness" in guarded.reason


def test_strict_mode_leaves_a_grounded_answer_alone(contexts, settings):
    answer = "Customers may request a refund within 30 days of purchase [1]."
    result = check_groundedness(
        answer, contexts, [cite(1, contexts[0])], [], strict(settings)
    )

    assert result.score == pytest.approx(1.0)
    assert result.abstained is False
    assert result.reason is None


@pytest.mark.parametrize("strict_mode", [False, True])
def test_the_abstention_message_counts_as_grounded(contexts, settings, strict_mode):
    cfg = strict(settings) if strict_mode else settings

    result = check_groundedness(cfg.abstain_message, contexts, [], [], cfg)

    assert result.abstained is True
    assert result.score == pytest.approx(1.0)
    assert result.unsupported == []
    assert result.reason is not None


def test_the_abstention_message_is_matched_after_normalisation(contexts, settings):
    noisy = f"  {settings.abstain_message.upper()}  "
    assert check_groundedness(noisy, contexts, [], [], settings).abstained is True


def test_abstention_still_reports_invalid_citations(contexts, settings):
    result = check_groundedness(settings.abstain_message, contexts, [], [4], settings)
    assert result.invalid_citations == [4]


def test_embedder_does_not_change_the_verdict_on_clear_cut_answers(
    contexts, settings, embedder
):
    citations = [cite(1, contexts[0])]
    grounded = "Customers may request a refund within 30 days of purchase [1]."
    invented = f"{FABRICATED} [1]."

    for answer, expected in ((grounded, 1.0), (invented, 0.0)):
        plain = check_groundedness(answer, contexts, citations, [], settings)
        with_embedder = check_groundedness(
            answer, contexts, citations, [], settings, embedder=embedder
        )
        assert plain.score == pytest.approx(expected)
        assert with_embedder.score == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# should_abstain
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "top_score,expected",
    [(-1.0, True), (0.0, True), (0.099, True), (0.10, False), (0.5, False), (1.0, False)],
)
def test_should_abstain_uses_the_minimum_retrieval_score(settings, top_score, expected):
    assert settings.min_retrieval_score == pytest.approx(0.10)
    assert should_abstain(top_score, settings) is expected


def test_should_abstain_follows_a_reconfigured_threshold(settings):
    picky = settings.model_copy(update={"min_retrieval_score": 0.8})
    assert should_abstain(0.5, picky) is True
    assert should_abstain(0.5, settings) is False


# --------------------------------------------------------------------------- #
# Pre-generation query coverage gate
# --------------------------------------------------------------------------- #
def test_query_coverage_is_one_when_every_term_is_present(scored_factory) -> None:
    contexts = [scored_factory("c1", "Approved refunds settle within 5 to 10 business days.")]
    assert query_term_coverage("Do approved refunds settle?", contexts) == 1.0


def test_query_coverage_counts_each_missing_term(scored_factory) -> None:
    contexts = [scored_factory("c1", "Approved refunds settle within 5 to 10 business days.")]
    # "long" is a content word the passage never uses: 3 of 4 terms covered.
    assert query_term_coverage("How long do approved refunds settle?", contexts) == 0.75


def test_query_coverage_falls_when_the_subject_is_absent(scored_factory) -> None:
    contexts = [scored_factory("c1", "Rideshare and taxi fares are reimbursed on expenses.")]
    # "cryptocurrency" and "payments" appear nowhere, which is the whole signal.
    assert query_term_coverage("What is the cryptocurrency payments policy?", contexts) < 0.5


def test_query_coverage_ignores_inflection(scored_factory) -> None:
    contexts = [scored_factory("c1", "The engineer must acknowledge a page within 5 minutes.")]
    assert query_term_coverage("Which engineers acknowledged the paging?", contexts) == 1.0


def test_query_coverage_of_a_contentless_question_defers_to_the_score_gate(
    scored_factory,
) -> None:
    assert query_term_coverage("Why?", [scored_factory("c1", "anything")]) == 1.0


def test_query_coverage_with_no_contexts_is_zero(scored_factory) -> None:
    assert query_term_coverage("cryptocurrency payments policy", []) == 0.0
