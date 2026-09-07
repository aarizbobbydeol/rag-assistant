"""Tests for the offline evaluation harness: metrics and golden-set plumbing.

Every expected value below is arithmetic done by hand and written out as a
literal, so a metric that changes definition fails here instead of silently
re-scoring an ablation sweep.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.errors import ConfigurationError
from app.models import Chunk
from eval.dataset import EvalExample, load_dataset, resolve_relevant_chunks, save_dataset
from eval.metrics import (
    aggregate,
    average_precision,
    citation_precision,
    citation_recall,
    exact_match,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rouge_l,
    token_f1,
)

# One fixed ranking used by the whole retrieval block.
#   rank:      1     2     3     4     5
#   id:       c1    c2    c3    c4    c5
#   relevant:  -   yes     -     -   yes      (plus c9, never retrieved)
RANKED = ["c1", "c2", "c3", "c4", "c5"]
RELEVANT = ["c2", "c5", "c9"]


# --------------------------------------------------------------------------- #
# Retrieval metrics on a known ranking
# --------------------------------------------------------------------------- #
def test_recall_at_k_counts_hits_over_the_relevant_set() -> None:
    assert recall_at_k(RANKED, RELEVANT, 1) == pytest.approx(0.0)
    # only c2 in the top 3, out of three relevant ids
    assert recall_at_k(RANKED, RELEVANT, 3) == pytest.approx(1 / 3)
    # c2 and c5 in the top 5; c9 was never retrieved
    assert recall_at_k(RANKED, RELEVANT, 5) == pytest.approx(2 / 3)
    assert recall_at_k(RANKED, RELEVANT, 50) == pytest.approx(2 / 3)


def test_precision_at_k_divides_by_the_retrieved_count() -> None:
    assert precision_at_k(RANKED, RELEVANT, 2) == pytest.approx(0.5)
    assert precision_at_k(RANKED, RELEVANT, 3) == pytest.approx(1 / 3)
    assert precision_at_k(RANKED, RELEVANT, 5) == pytest.approx(0.4)
    # k beyond the ranking must not dilute the score with phantom misses
    assert precision_at_k(RANKED, RELEVANT, 10) == pytest.approx(0.4)


def test_hit_rate_is_binary_at_the_first_relevant_rank() -> None:
    assert hit_rate_at_k(RANKED, RELEVANT, 1) == pytest.approx(0.0)
    assert hit_rate_at_k(RANKED, RELEVANT, 2) == pytest.approx(1.0)
    assert hit_rate_at_k(RANKED, RELEVANT, 5) == pytest.approx(1.0)


def test_mrr_uses_the_first_relevant_rank() -> None:
    assert mrr(RANKED, RELEVANT) == pytest.approx(0.5)
    assert mrr(RANKED, ["c1"]) == pytest.approx(1.0)
    assert mrr(RANKED, ["c4"]) == pytest.approx(0.25)
    assert mrr(RANKED, ["c9"]) == pytest.approx(0.0)


def test_ndcg_at_k_against_hand_computed_dcg_and_idcg() -> None:
    # DCG@5 = 1/log2(3) + 1/log2(6) = 0.6309297535714575 + 0.3868528072345416
    #       = 1.0177825608059992
    # IDCG  = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1 + 0.6309297535714575 + 0.5
    #       = 2.1309297535714578      (three relevant ids, k = 5)
    assert ndcg_at_k(RANKED, RELEVANT, 5) == pytest.approx(0.4776237035032179)
    # DCG@3 = 1/log2(3) only; the ideal at k=3 is still all three relevant ids
    assert ndcg_at_k(RANKED, RELEVANT, 3) == pytest.approx(0.2960819109658652)
    # a perfect prefix scores 1.0 even though the relevant set is larger than k
    assert ndcg_at_k(["c2", "c5"], RELEVANT, 2) == pytest.approx(1.0)


def test_average_precision_is_normalised_by_the_whole_relevant_set() -> None:
    # hits at rank 2 (precision 1/2) and rank 5 (precision 2/5), over 3 relevant
    # ids: (0.5 + 0.4) / 3 = 0.30000000000000004
    assert average_precision(RANKED, RELEVANT) == pytest.approx(0.3)
    assert average_precision(RANKED, ["c1", "c2"]) == pytest.approx(1.0)


def test_duplicate_ids_in_a_ranking_are_not_credited_twice() -> None:
    assert precision_at_k(["c2", "c2", "c3"], ["c2"], 3) == pytest.approx(0.5)
    assert recall_at_k(["c2", "c2"], ["c2", "c5"], 2) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "metric",
    [recall_at_k, precision_at_k, hit_rate_at_k, ndcg_at_k],
)
def test_at_k_metrics_survive_degenerate_input(metric) -> None:
    assert metric([], RELEVANT, 5) == 0.0
    assert metric(RANKED, [], 5) == 0.0
    assert metric([], [], 5) == 0.0
    assert metric(RANKED, RELEVANT, 0) == 0.0
    assert metric(RANKED, RELEVANT, -1) == 0.0


@pytest.mark.parametrize("metric", [mrr, average_precision, citation_precision, citation_recall])
def test_unbounded_metrics_survive_degenerate_input(metric) -> None:
    assert metric([], RELEVANT) == 0.0
    assert metric(RANKED, []) == 0.0
    assert metric([], []) == 0.0


# --------------------------------------------------------------------------- #
# Citation metrics
# --------------------------------------------------------------------------- #
def test_citation_precision_and_recall_are_asymmetric() -> None:
    cited = ["c1", "c2"]
    # one of two citations is relevant; one of three relevant chunks was cited
    assert citation_precision(cited, RELEVANT) == pytest.approx(0.5)
    assert citation_recall(cited, RELEVANT) == pytest.approx(1 / 3)
    assert citation_precision(["c2", "c5", "c9"], RELEVANT) == pytest.approx(1.0)
    assert citation_recall(["c2", "c5", "c9"], RELEVANT) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Answer metrics
# --------------------------------------------------------------------------- #
def test_token_f1_on_a_known_pair() -> None:
    # "the cat sat on the mat" vs "the cat is on the mat": six tokens each,
    # multiset overlap {the x2, cat, on, mat} = 5 -> P = R = F1 = 5/6
    assert token_f1("the cat sat on the mat", "the cat is on the mat") == pytest.approx(
        0.8333333333333334
    )
    assert token_f1("Refunds take 5 days.", "refunds take 5 days") == pytest.approx(1.0)
    assert token_f1("totally unrelated wording", "refund policy") == pytest.approx(0.0)


def test_rouge_l_is_order_sensitive_where_token_f1_is_not() -> None:
    # identical bags, reversed order: F1 sees a perfect match, ROUGE-L sees an
    # LCS of length 1 -> P = R = 1/4 -> F1 = 0.25
    assert token_f1("a b c d", "d c b a") == pytest.approx(1.0)
    assert rouge_l("a b c d", "d c b a") == pytest.approx(0.25)
    # LCS of "the cat sat on the mat" / "the cat is on the mat" is 5 tokens
    assert rouge_l("the cat sat on the mat", "the cat is on the mat") == pytest.approx(
        0.8333333333333334
    )
    # LCS 2 of 4 predicted and 2 reference tokens -> P = 0.5, R = 1.0, F1 = 2/3
    assert rouge_l("a x b y", "a b") == pytest.approx(2 / 3)


def test_exact_match_ignores_case_and_punctuation() -> None:
    assert exact_match("The Cat sat.", "the cat sat") == pytest.approx(1.0)
    assert exact_match("the cat sat", "the cat ran") == pytest.approx(0.0)
    # both sides empty is the abstention case, and counts as agreement
    assert exact_match("", "") == pytest.approx(1.0)
    assert exact_match("", "an answer") == pytest.approx(0.0)


@pytest.mark.parametrize("metric", [token_f1, rouge_l])
def test_text_metrics_return_zero_not_nan_on_empty_input(metric) -> None:
    assert metric("", "reference text") == 0.0
    assert metric("prediction text", "") == 0.0
    assert metric("", "") == 0.0
    assert metric("...", "!!!") == 0.0  # tokenises to nothing


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def test_aggregate_reports_means_stdevs_and_n() -> None:
    rows = [
        {"question": "q1", "recall": 1.0, "mrr": 0.5, "grounded": True},
        {"question": "q2", "recall": 0.0, "mrr": 1.0, "grounded": False},
    ]
    summary = aggregate(rows)
    assert summary["n"] == 2
    assert summary["recall"] == pytest.approx(0.5)
    assert summary["mrr"] == pytest.approx(0.75)
    # sample stdev of [1, 0] is sqrt(0.5); of [0.5, 1.0] is sqrt(0.125)
    assert summary["recall_stdev"] == pytest.approx(0.7071067811865476)
    assert summary["mrr_stdev"] == pytest.approx(0.3535533905932738)
    # booleans aggregate as rates, strings are dropped entirely
    assert summary["grounded"] == pytest.approx(0.5)
    assert "question" not in summary


def test_aggregate_handles_empty_single_and_ragged_rows() -> None:
    assert aggregate([]) == {"n": 0}

    single = aggregate([{"recall": 0.25}])
    assert single["recall"] == pytest.approx(0.25)
    assert single["recall_stdev"] == 0.0

    ragged = aggregate([{"recall": 1.0}, {"mrr": 0.5}])
    assert ragged["n"] == 2
    assert ragged["recall"] == pytest.approx(1.0)
    assert ragged["mrr"] == pytest.approx(0.5)


def test_aggregate_drops_non_finite_values() -> None:
    summary = aggregate([{"score": 1.0}, {"score": float("nan")}, {"score": float("inf")}])
    assert summary["score"] == pytest.approx(1.0)
    assert summary["score_stdev"] == 0.0


# --------------------------------------------------------------------------- #
# Golden-set I/O
# --------------------------------------------------------------------------- #
def _example(question: str = "What is the refund window?") -> EvalExample:
    return EvalExample(
        question=question,
        answer="Thirty days from purchase.",
        relevant_doc_ids=["doc-refund"],
        relevant_chunk_texts=["Customers may request a refund within 30 days of purchase."],
        tags=["policy"],
    )


def test_dataset_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    examples = [_example(), EvalExample("Who won?", "", [], [], must_abstain=True)]

    assert save_dataset(path, examples) == 2
    assert load_dataset(path) == examples


def test_load_dataset_tolerates_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(
        "\n"
        '{"question": "a", "answer": "A", "relevant_doc_ids": ["d1"], '
        '"relevant_chunk_texts": []}\n'
        "   \n"
        '{"question": "b"}\n'
        "\n",
        encoding="utf-8",
    )
    examples = load_dataset(path)
    assert [item.question for item in examples] == ["a", "b"]
    # omitted fields fall back to empty rather than exploding
    assert examples[1].answer == ""
    assert examples[1].relevant_doc_ids == []
    assert examples[1].must_abstain is False


def test_load_dataset_names_the_offending_line(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text(
        '{"question": "a", "answer": "A", "relevant_doc_ids": [], '
        '"relevant_chunk_texts": []}\n'
        "\n"
        '{"question": "b", "answer":}\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError) as excinfo:
        load_dataset(path)
    assert "line 3" in str(excinfo.value)
    assert "broken.jsonl" in str(excinfo.value)


def test_load_dataset_rejects_an_example_without_a_question(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text('{"answer": "A"}\n', encoding="utf-8")
    with pytest.raises(ConfigurationError) as excinfo:
        load_dataset(path)
    assert "line 1" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# resolve_relevant_chunks - the reason the golden set survives re-chunking
# --------------------------------------------------------------------------- #
PASSAGE = (
    "Customers may request a refund within 30 days of purchase. "
    "Refunds are issued to the original payment method."
)


def test_resolve_matches_the_same_passage_across_two_chunkings(chunk_factory) -> None:
    example = EvalExample(
        question="How long is the refund window?",
        answer="30 days.",
        relevant_doc_ids=["doc-1"],
        relevant_chunk_texts=[PASSAGE],
    )

    # Chunking A: one coarse chunk that swallows the passage whole.
    coarse = [
        chunk_factory("a1", "# Refund policy\n\n" + PASSAGE + "\n\nContact support to start.", 0),
        chunk_factory("a2", "Digital downloads are non-refundable once accessed.", 1),
    ]
    # Chunking B: sentence-level chunks, reflowed whitespace and different case.
    fine = [
        chunk_factory("b1", "Customers may request a  refund\nwithin 30 days of purchase.", 0),
        chunk_factory("b2", "REFUNDS ARE ISSUED TO THE ORIGINAL PAYMENT METHOD.", 1),
        chunk_factory("b3", "Shipping fees are never refunded.", 2),
    ]

    assert resolve_relevant_chunks(example, coarse) == {"a1"}
    assert resolve_relevant_chunks(example, fine) == {"b1", "b2"}


def test_resolve_falls_back_to_doc_ids_when_no_text_matches(chunk_factory) -> None:
    example = EvalExample(
        question="Unanswerable from the chunks below",
        answer="",
        relevant_doc_ids=["doc-1"],
        relevant_chunk_texts=["lattice gauge quantum chromodynamics simulation"],
    )
    chunks = [
        chunk_factory("c1", "Refunds settle within 5 to 10 business days.", 0),
        chunk_factory("c2", "Shipping fees are never refunded.", 1),
        chunk_factory("c1-other", "Unrelated document text.", 0).model_copy(
            update={"doc_id": "doc-2"}
        ),
    ]
    assert resolve_relevant_chunks(example, chunks) == {"c1", "c2"}


def test_resolve_uses_doc_ids_when_no_passages_are_recorded(chunk_factory) -> None:
    example = EvalExample("Q", "A", ["doc-1"], [])
    chunks = [chunk_factory("c1", "Some text.", 0)]
    assert resolve_relevant_chunks(example, chunks) == {"c1"}


def test_resolve_returns_empty_when_nothing_can_be_resolved(chunk_factory) -> None:
    example = EvalExample("Q", "A", [], ["nothing like this appears anywhere"])
    chunks = [chunk_factory("c1", "Refunds settle within 5 to 10 business days.", 0)]
    assert resolve_relevant_chunks(example, chunks) == set()
    assert resolve_relevant_chunks(example, []) == set()


def test_resolve_recovers_a_passage_split_across_chunk_boundaries(chunk_factory) -> None:
    """No chunk holds the whole passage, but one holds most of its words."""
    example = EvalExample(
        question="How long is the refund window?",
        answer="30 days.",
        relevant_doc_ids=[],
        relevant_chunk_texts=[PASSAGE],
    )
    chunks: list[Chunk] = [
        chunk_factory(
            "s1",
            "Customers may request a refund within 30 days of the purchase, "
            "and refunds are then issued to the original method of payment.",
            0,
        ),
        chunk_factory("s2", "Support hours are 9am to 5pm on weekdays.", 1),
    ]
    assert resolve_relevant_chunks(example, chunks) == {"s1"}
