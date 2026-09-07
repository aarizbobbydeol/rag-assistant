"""BM25 index: ranking quality, incremental bookkeeping and round-tripping."""

from __future__ import annotations

import json

import pytest

from app.retrieval.bm25 import BM25Index

CORPUS = {
    "refunds": "Customers may request a refund within thirty days of purchase.",
    "shipping": "Shipping fees are never refunded and shipping is charged per order.",
    "downloads": "Digital downloads are non refundable once accessed.",
    "support": "Contact support by email for any question about an order.",
}


def build(**kwargs: float) -> BM25Index:
    index = BM25Index(**kwargs)
    index.add(list(CORPUS), list(CORPUS.values()))
    return index


def test_empty_index_returns_nothing():
    index = BM25Index()
    assert len(index) == 0
    assert index.search("refund", 5) == []


def test_empty_and_punctuation_only_query_returns_nothing():
    index = build()
    assert index.search("", 5) == []
    assert index.search("   ...   ", 5) == []


def test_zero_k_returns_nothing():
    assert build().search("refund", 0) == []


def test_k_larger_than_matches_is_clamped():
    index = build()
    hits = index.search("refund", 50)
    # No stemming: only the document with the literal token "refund" matches,
    # so k is bounded by the postings list, not by the corpus size.
    assert [doc_id for doc_id, _ in hits] == ["refunds"]


def test_ranks_the_document_about_the_query_first():
    index = build()
    hits = index.search("refund within thirty days", 3)
    assert hits[0][0] == "refunds"
    assert all(score > 0.0 for _, score in hits)


def test_scores_are_descending():
    index = build()
    scores = [score for _, score in index.search("shipping order refund", 4)]
    assert scores == sorted(scores, reverse=True)


def test_rare_term_outweighs_a_common_one():
    index = BM25Index()
    index.add(
        ["a", "b", "c", "d"],
        [
            "alpha common common",
            "beta common common",
            "gamma common common",
            "delta rare common",
        ],
    )
    # "rare" appears in one document, "common" in all four; the document holding
    # the rare term must win even though both queries are single terms.
    assert index.search("rare", 4)[0][0] == "d"
    assert index.search("rare common", 4)[0][0] == "d"


def test_length_normalisation_prefers_the_shorter_document():
    index = BM25Index()
    index.add(
        ["short", "long"],
        ["refund policy", "refund policy " + " ".join(f"filler{n}" for n in range(200))],
    )
    hits = dict(index.search("refund policy", 2))
    assert hits["short"] > hits["long"]


def test_ties_break_by_id_ascending():
    index = BM25Index()
    index.add(["zebra", "alpha", "mango"], ["identical text"] * 3)
    assert [doc_id for doc_id, _ in index.search("identical text", 3)] == [
        "alpha",
        "mango",
        "zebra",
    ]


def test_incremental_add_matches_a_single_batch():
    incremental = BM25Index()
    for doc_id, text in CORPUS.items():
        incremental.add([doc_id], [text])
    assert incremental.search("refund order", 4) == build().search("refund order", 4)


def test_average_length_and_df_survive_deletion():
    full = build()
    assert full.delete(["shipping", "missing"]) == 1
    assert len(full) == 3

    rebuilt = BM25Index()
    remaining = {k: v for k, v in CORPUS.items() if k != "shipping"}
    rebuilt.add(list(remaining), list(remaining.values()))

    assert full.avg_doc_length == pytest.approx(rebuilt.avg_doc_length)
    assert full.search("refund shipping order", 4) == rebuilt.search("refund shipping order", 4)
    assert all(doc_id != "shipping" for doc_id, _ in full.search("shipping", 4))


def test_readding_an_id_replaces_the_old_text():
    index = build()
    index.add(["support"], ["refund refund refund refund refund"])
    assert len(index) == 4
    hits = dict(index.search("support", 4))
    assert "support" not in hits
    assert index.search("refund", 4)[0][0] == "support"


def test_delete_then_readd_is_equivalent_to_never_deleting():
    index = build()
    index.delete(["downloads"])
    index.add(["downloads"], [CORPUS["downloads"]])
    assert index.search("refund downloads order", 4) == build().search(
        "refund downloads order", 4
    )


def test_clear_empties_the_index():
    index = build()
    index.clear()
    assert len(index) == 0
    assert index.avg_doc_length == 0.0
    assert index.search("refund", 3) == []


def test_to_dict_from_dict_round_trip():
    index = build(k1=1.2, b=0.6)
    restored = BM25Index.from_dict(json.loads(json.dumps(index.to_dict())))

    assert (restored.k1, restored.b) == (1.2, 0.6)
    assert len(restored) == len(index)
    assert restored.avg_doc_length == pytest.approx(index.avg_doc_length)
    for query in ("refund", "shipping order", "digital downloads"):
        assert restored.search(query, 4) == index.search(query, 4)


def test_round_trip_stays_writable():
    restored = BM25Index.from_dict(build().to_dict())
    restored.add(["late"], ["A late refund request is still a refund request."])
    assert restored.delete(["support"]) == 1
    assert len(restored) == 4
    assert restored.search("late refund", 4)[0][0] == "late"


def test_mismatched_ids_and_texts_are_rejected():
    with pytest.raises(ValueError):
        BM25Index().add(["a", "b"], ["only one text"])


def test_empty_document_is_indexed_but_never_matches():
    index = BM25Index()
    index.add(["blank", "real"], ["", "refund policy"])
    assert len(index) == 2
    assert [doc_id for doc_id, _ in index.search("refund", 2)] == ["real"]
