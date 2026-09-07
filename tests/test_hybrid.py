"""Fusion and MMR: arithmetic, edge cases, and one end-to-end retrieval check."""

from __future__ import annotations

import numpy as np
import pytest

from app.retrieval.bm25 import BM25Index
from app.retrieval.hybrid import (
    min_max_normalise,
    mmr_select,
    reciprocal_rank_fusion,
    weighted_fusion,
)


# --------------------------------------------------------------------------- #
# min_max_normalise
# --------------------------------------------------------------------------- #
def test_min_max_normalise_maps_onto_unit_interval():
    assert min_max_normalise([2.0, 4.0, 6.0]) == [0.0, 0.5, 1.0]


def test_min_max_normalise_handles_degenerate_inputs():
    assert min_max_normalise([]) == []
    assert min_max_normalise([7.5]) == [0.0]
    assert min_max_normalise([3.0, 3.0, 3.0]) == [0.0, 0.0, 0.0]
    assert min_max_normalise([-1.0, 0.0, 1.0]) == [0.0, 0.5, 1.0]


# --------------------------------------------------------------------------- #
# reciprocal_rank_fusion
# --------------------------------------------------------------------------- #
def test_rrf_scores_are_the_sum_of_reciprocal_ranks():
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_accepts_scored_pairs_and_ignores_the_scores():
    pairs = reciprocal_rank_fusion([[("a", 99.0), ("b", 0.1)], [("b", 0.2), ("a", 0.1)]])
    bare = reciprocal_rank_fusion([["a", "b"], ["b", "a"]])
    assert pairs == bare


def test_rrf_rewards_agreement_over_a_single_strong_hit():
    dense = ["solo", "agreed", "filler1", "filler2"]
    lexical = ["other", "agreed", "filler2", "filler1"]
    ranked = [doc_id for doc_id, _ in reciprocal_rank_fusion([dense, lexical], k=5)]
    assert ranked[0] == "agreed"


def test_rrf_weights_shift_the_winner():
    dense = ["d", "shared"]
    lexical = ["l", "shared"]
    assert reciprocal_rank_fusion([dense, lexical], k=1, weights=[5.0, 0.1])[0][0] == "d"
    assert reciprocal_rank_fusion([dense, lexical], k=1, weights=[0.1, 5.0])[0][0] == "l"


def test_rrf_ties_break_by_id():
    ranked = reciprocal_rank_fusion([["zeta", "alpha"], ["alpha", "zeta"]])
    assert [doc_id for doc_id, _ in ranked] == ["alpha", "zeta"]


def test_rrf_handles_empty_input():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
    assert [doc_id for doc_id, _ in reciprocal_rank_fusion([[], ["a"]])] == ["a"]


def test_rrf_collapses_a_repeated_id_within_one_ranking():
    # A duplicate is dropped rather than left as a gap, so what follows it moves
    # up: "b" is rank 2, not rank 3, and "a" is only counted once.
    fused = dict(reciprocal_rank_fusion([["a", "a", "b"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62)


def test_rrf_rejects_mismatched_weights():
    with pytest.raises(ValueError, match="weights"):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])


def test_rrf_rejects_a_damping_constant_below_one():
    with pytest.raises(ValueError, match="k must be"):
        reciprocal_rank_fusion([["a"]], k=0)


# --------------------------------------------------------------------------- #
# weighted_fusion
# --------------------------------------------------------------------------- #
def test_weighted_fusion_blends_normalised_scores():
    dense = [("a", 1.0), ("b", 0.0)]
    lexical = [("b", 10.0), ("a", 0.0)]
    fused = dict(weighted_fusion(dense, lexical, 0.75, 0.25))
    assert fused["a"] == pytest.approx(0.75)
    assert fused["b"] == pytest.approx(0.25)


def test_weighted_fusion_respects_the_weights():
    dense = [("a", 1.0), ("b", 0.0)]
    lexical = [("b", 1.0), ("a", 0.0)]
    assert weighted_fusion(dense, lexical, 0.9, 0.1)[0][0] == "a"
    assert weighted_fusion(dense, lexical, 0.1, 0.9)[0][0] == "b"


def test_weighted_fusion_survives_single_element_and_constant_lists():
    single = weighted_fusion([("a", 5.0)], [("b", 5.0)], 0.5, 0.5)
    assert dict(single) == {"a": 0.0, "b": 0.0}

    constant = weighted_fusion([("a", 2.0), ("b", 2.0)], [], 0.5, 0.5)
    assert dict(constant) == {"a": 0.0, "b": 0.0}


def test_weighted_fusion_handles_empty_inputs_and_ties():
    assert weighted_fusion([], [], 0.5, 0.5) == []
    ranked = weighted_fusion([("zeta", 1.0), ("alpha", 1.0)], [], 1.0, 0.0)
    assert [doc_id for doc_id, _ in ranked] == ["alpha", "zeta"]


# --------------------------------------------------------------------------- #
# mmr_select
# --------------------------------------------------------------------------- #
def _mmr_fixture() -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Three near-duplicates plus one outlier, in descending relevance order."""
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.14, 0.0],
            [0.98, 0.20, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    scores = [0.90, 0.85, 0.80, 0.40]
    return np.array([1.0, 0.0, 0.0], dtype=np.float32), vectors, scores


def test_mmr_lambda_one_is_pure_relevance_order():
    query, vectors, scores = _mmr_fixture()
    assert mmr_select(query, vectors, scores, k=4, lambda_mult=1.0) == [0, 1, 2, 3]


def test_mmr_low_lambda_pulls_in_the_diverse_candidate():
    query, vectors, scores = _mmr_fixture()
    diverse = mmr_select(query, vectors, scores, k=2, lambda_mult=0.2)
    # The outlier is the least relevant candidate but the only novel one, so a
    # diversity-weighted selection must reach past both near-duplicates for it.
    assert diverse == [0, 3]
    assert mmr_select(query, vectors, scores, k=2, lambda_mult=1.0) == [0, 1]


def test_mmr_always_starts_with_the_most_relevant_candidate():
    query, vectors, scores = _mmr_fixture()
    for lambda_mult in (0.0, 0.3, 0.7, 1.0):
        assert mmr_select(query, vectors, scores, k=3, lambda_mult=lambda_mult)[0] == 0


def test_mmr_handles_k_larger_than_the_candidate_set():
    query, vectors, scores = _mmr_fixture()
    selected = mmr_select(query, vectors, scores, k=99, lambda_mult=0.7)
    assert sorted(selected) == [0, 1, 2, 3]


def test_mmr_handles_empty_and_zero_k():
    query, vectors, scores = _mmr_fixture()
    assert mmr_select(query, np.zeros((0, 3), dtype=np.float32), [], k=5) == []
    assert mmr_select(query, [], [], k=5) == []
    assert mmr_select(query, vectors, scores, k=0) == []


def test_mmr_falls_back_to_the_query_vector_when_scores_are_missing():
    query, vectors, _ = _mmr_fixture()
    assert mmr_select(query, vectors, [], k=4, lambda_mult=1.0) == [0, 1, 2, 3]


def test_mmr_tolerates_unnormalised_and_zero_vectors():
    vectors = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    selected = mmr_select(np.array([1.0, 0.0]), vectors, [0.1, 0.9, 0.5], k=3, lambda_mult=0.5)
    assert sorted(selected) == [0, 1, 2]
    assert selected[0] == 1


# --------------------------------------------------------------------------- #
# Behavioural: BM25 and dense disagree, RRF surfaces the consensus document
# --------------------------------------------------------------------------- #
QUERY = "python memory management"

CORPUS: dict[str, str] = {
    # Keyword soup: unbeatable on BM25, nearly worthless as an answer.
    "stuffed": "Python memory management quiz: python memory management memory python.",
    # The document a human would pick: on-topic wording *and* on-topic meaning.
    "gc": (
        "Python manages memory automatically. The memory held by unreachable "
        "objects is reclaimed by the garbage collector."
    ),
    # Says the same thing in different words, so BM25 cannot see it at all.
    "cycles": (
        "Reference counting plus a cycle detector frees storage held by objects "
        "that nothing points at any more."
    ),
    "threads": "Python threading and the global interpreter lock, explained with examples.",
}

# Hand-built topic vectors: [reclamation, concurrency, quiz-chatter].
DENSE_VECTORS: dict[str, list[float]] = {
    "stuffed": [0.10, 0.10, 0.99],
    "gc": [0.95, 0.20, 0.10],
    "cycles": [0.99, 0.10, 0.00],
    "threads": [0.30, 0.90, 0.10],
}
QUERY_VECTOR = [1.0, 0.0, 0.0]


def _dense_ranking() -> list[tuple[str, float]]:
    """Cosine ranking over the hand-built vectors - no embedder dependency."""
    query = np.array(QUERY_VECTOR, dtype=np.float64)
    query /= np.linalg.norm(query)
    scored = []
    for doc_id, raw in DENSE_VECTORS.items():
        vector = np.array(raw, dtype=np.float64)
        scored.append((doc_id, float(query @ vector / np.linalg.norm(vector))))
    return sorted(scored, key=lambda item: (-item[1], item[0]))


def test_rrf_surfaces_the_document_both_retrievers_agree_on():
    index = BM25Index()
    index.add(list(CORPUS), list(CORPUS.values()))
    lexical = index.search(QUERY, 10)
    dense = _dense_ranking()

    lexical_ids = [doc_id for doc_id, _ in lexical]
    dense_ids = [doc_id for doc_id, _ in dense]

    # The premise: the two retrievers genuinely disagree about the best answer.
    assert lexical_ids[0] == "stuffed"
    assert dense_ids[0] == "cycles"
    assert "cycles" not in lexical_ids  # shares no vocabulary with the query
    assert dense_ids[-1] == "stuffed"  # semantically the worst match

    fused = reciprocal_rank_fusion([dense, lexical], k=60)
    fused_ids = [doc_id for doc_id, _ in fused]

    # Neither retriever's winner survives; the document both put second does.
    assert fused_ids[0] == "gc"
    assert dict(fused)["gc"] == pytest.approx(2 / 62)


def test_weighted_fusion_can_be_tilted_to_either_retriever():
    index = BM25Index()
    index.add(list(CORPUS), list(CORPUS.values()))
    lexical = index.search(QUERY, 10)
    dense = _dense_ranking()

    assert weighted_fusion(dense, lexical, 1.0, 0.0)[0][0] == "cycles"
    assert weighted_fusion(dense, lexical, 0.0, 1.0)[0][0] == "stuffed"
