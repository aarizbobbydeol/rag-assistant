"""Fusing the dense and lexical result lists, and diversifying what survives.

Reciprocal rank fusion is the default because it only needs the *order* of each
retriever's output: cosine similarities and BM25 scores live on incomparable
scales, and any attempt to blend the raw numbers ends up tuned to one corpus.
`weighted_fusion` is kept for the ablation sweep, which does want the scores.

Every function here is on the scoring path, so every tie is broken explicitly
by id or by index - never by set iteration order or by whatever `sorted` felt
like doing with equal keys.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Each retriever may hand us bare ids or the `(id, score)` pairs its `search`
# returned; only the position matters to RRF, so accept both.
Ranking = Sequence[str | tuple[str, float]]


def _ranked_ids(ranking: Ranking) -> list[str]:
    """Ids in rank order, first occurrence winning if a retriever repeats one."""
    seen: set[str] = set()
    ordered: list[str] = []
    for item in ranking:
        doc_id = item if isinstance(item, str) else str(item[0])
        if doc_id not in seen:
            seen.add(doc_id)
            ordered.append(doc_id)
    return ordered


def reciprocal_rank_fusion(
    rankings: Sequence[Ranking],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked lists as ``sum(w_i / (k + rank_i))`` over the lists an id is in.

    Ranks are 1-based. ``k`` damps the head of each list so that a single
    retriever's top hit cannot dominate a document that both retrievers like -
    which is the whole point of running two of them.
    """
    if not rankings:
        return []
    if k < 1:
        raise ValueError(f"rrf k must be >= 1, got {k}")
    if weights is None:
        weights = [1.0] * len(rankings)
    elif len(weights) != len(rankings):
        raise ValueError(
            f"weights has {len(weights)} entries but there are {len(rankings)} rankings"
        )

    fused: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, doc_id in enumerate(_ranked_ids(ranking), start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + float(weight) / (k + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))


def min_max_normalise(scores: Sequence[float]) -> list[float]:
    """Map ``scores`` onto ``[0, 1]``; all-equal (or single) input maps to 0.0.

    Collapsing a constant list to zero rather than one is deliberate: a
    retriever that cannot separate its own candidates should not out-vote one
    that can.
    """
    values = [float(score) for score in scores]
    if not values:
        return []
    lowest = min(values)
    span = max(values) - lowest
    if span <= 0.0:
        return [0.0] * len(values)
    return [(value - lowest) / span for value in values]


def _best_per_id(results: Sequence[tuple[str, float]]) -> dict[str, float]:
    normalised = min_max_normalise([score for _, score in results])
    best: dict[str, float] = {}
    for (doc_id, _), value in zip(results, normalised, strict=True):
        if value > best.get(doc_id, float("-inf")):
            best[doc_id] = value
    return best


def weighted_fusion(
    dense: Sequence[tuple[str, float]],
    lexical: Sequence[tuple[str, float]],
    dense_weight: float,
    lexical_weight: float,
) -> list[tuple[str, float]]:
    """Blend two scored lists after min-max normalising each one separately.

    A document missing from one list contributes nothing from that side rather
    than a penalty, so a strong single-retriever hit still survives fusion.
    """
    fused: dict[str, float] = {}
    for results, weight in ((dense, dense_weight), (lexical, lexical_weight)):
        for doc_id, value in _best_per_id(results).items():
            fused[doc_id] = fused.get(doc_id, 0.0) + float(weight) * value
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0.0, 1.0, norms)


def mmr_select(
    query_vec: np.ndarray,
    candidate_vecs: np.ndarray,
    candidate_scores: Sequence[float],
    k: int,
    lambda_mult: float = 0.7,
) -> list[int]:
    """Greedy Maximal Marginal Relevance; returns indices into the candidates.

    Each step picks ``argmax(lambda * relevance - (1 - lambda) * max similarity
    to what is already selected)``. Relevance is the retriever's own score,
    min-max normalised so it shares the ``[0, 1]`` scale of the cosine penalty -
    that rescaling is order-preserving, so ``lambda_mult == 1.0`` still returns
    the candidates in pure relevance order.

    ``query_vec`` is the fallback source of relevance: when ``candidate_scores``
    does not line up with the candidate matrix (the caller has vectors but no
    scores) relevance is recomputed as cosine against the query.
    """
    vectors = np.asarray(candidate_vecs, dtype=np.float32)
    if vectors.size == 0:
        return []
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    count = vectors.shape[0]
    k = min(int(k), count)
    if k <= 0:
        return []

    unit = _unit_rows(vectors)
    scores = [float(score) for score in candidate_scores]
    if len(scores) != count:
        query = _unit_rows(np.asarray(query_vec, dtype=np.float32).reshape(1, -1))[0]
        scores = (unit @ query).tolist()
    relevance = min_max_normalise(scores)
    similarity = unit @ unit.T

    selected: list[int] = []
    remaining = list(range(count))
    while remaining and len(selected) < k:
        best_index = remaining[0]
        best_value = float("-inf")
        # Ascending scan with a strict `>` makes the lowest index win a tie.
        for index in remaining:
            penalty = float(similarity[index, selected].max()) if selected else 0.0
            value = lambda_mult * relevance[index] - (1.0 - lambda_mult) * penalty
            if value > best_value:
                best_value = value
                best_index = index
        selected.append(best_index)
        remaining.remove(best_index)
    return selected
