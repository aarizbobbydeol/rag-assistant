"""Retrieval and answer quality metrics.

Deliberately pure: every function takes plain ids or plain strings and returns a
float in ``[0, 1]``. Nothing here imports the pipeline, so the harness, the
ablation sweep and the tests can all call the same arithmetic without building
an index first.

Two conventions hold everywhere:

* Degenerate input (empty ranking, no relevant documents, ``k <= 0``) scores
  ``0.0``. A metric is never allowed to raise or to return ``NaN``, because a
  single bad row would otherwise poison a whole sweep's mean.
* Rankings are de-duplicated, keeping the first occurrence. A retriever that
  returns the same chunk twice should not be credited twice.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Iterable, Sequence

from app.utils import tokenize

__all__ = [
    "recall_at_k",
    "precision_at_k",
    "hit_rate_at_k",
    "mrr",
    "ndcg_at_k",
    "average_precision",
    "citation_precision",
    "citation_recall",
    "token_f1",
    "rouge_l",
    "exact_match",
    "aggregate",
]


def _dedupe(ranked_ids: Sequence[str]) -> list[str]:
    """Order-preserving de-duplication (``dict`` keeps insertion order)."""
    return list(dict.fromkeys(ranked_ids))


def _prepare(
    ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int | None = None
) -> tuple[list[str], set[str]]:
    ranked = _dedupe(ranked_ids)
    if k is not None:
        ranked = ranked[: max(0, k)]
    return ranked, set(relevant_ids)


# --------------------------------------------------------------------------- #
# Retrieval metrics
# --------------------------------------------------------------------------- #
def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Fraction of the relevant set that appears in the top ``k``."""
    top, relevant = _prepare(ranked_ids, relevant_ids, k)
    if not top or not relevant:
        return 0.0
    return len(relevant.intersection(top)) / len(relevant)


def precision_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Fraction of the top ``k`` that is relevant.

    The denominator is ``min(k, len(ranked_ids))`` rather than ``k``: when the
    index holds fewer than ``k`` chunks, a perfect retriever would otherwise
    look worse than it is, which would distort comparisons between sweep
    configurations that return different candidate counts.
    """
    top, relevant = _prepare(ranked_ids, relevant_ids, k)
    if not top or not relevant:
        return 0.0
    return len(relevant.intersection(top)) / len(top)


def hit_rate_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """1.0 when at least one relevant id is in the top ``k``."""
    top, relevant = _prepare(ranked_ids, relevant_ids, k)
    if not top or not relevant:
        return 0.0
    return 1.0 if relevant.intersection(top) else 0.0


def mrr(ranked_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    """Reciprocal of the 1-based rank of the first relevant id."""
    ranked, relevant = _prepare(ranked_ids, relevant_ids)
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Binary-relevance nDCG with a ``1 / log2(rank + 1)`` discount.

    The ideal ranking holds ``min(k, len(relevant))`` relevant items, so an
    example with more relevant chunks than ``k`` can still reach 1.0 - the
    metric grades the ordering, not the cut-off.
    """
    top, relevant = _prepare(ranked_ids, relevant_ids, k)
    if not top or not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(top, start=1)
        if doc_id in relevant
    )
    ideal_hits = min(max(0, k), len(relevant))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def average_precision(ranked_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    """Mean of the precisions measured at each relevant hit.

    Normalised by the size of the whole relevant set, so relevant chunks that
    were never retrieved still cost the score.
    """
    ranked, relevant = _prepare(ranked_ids, relevant_ids)
    if not ranked or not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            hits += 1
            total += hits / rank
    return total / len(relevant)


# --------------------------------------------------------------------------- #
# Citation metrics
# --------------------------------------------------------------------------- #
def citation_precision(cited_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    """Share of the answer's citations that point at a genuinely relevant chunk."""
    cited, relevant = _prepare(cited_ids, relevant_ids)
    if not cited or not relevant:
        return 0.0
    return len(relevant.intersection(cited)) / len(cited)


def citation_recall(cited_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    """Share of the relevant chunks the answer actually cited."""
    cited, relevant = _prepare(cited_ids, relevant_ids)
    if not cited or not relevant:
        return 0.0
    return len(relevant.intersection(cited)) / len(relevant)


# --------------------------------------------------------------------------- #
# Answer metrics
# --------------------------------------------------------------------------- #
def token_f1(prediction: str, reference: str) -> float:
    """Bag-of-tokens F1 (the SQuAD measure) over :func:`app.utils.tokenize`.

    Multiset intersection, so a word the reference repeats only counts twice if
    the prediction repeats it too. Returns 0.0 when either side has no tokens:
    F1 is undefined there, and 0.0 keeps the aggregate finite.
    """
    pred_tokens = Counter(tokenize(prediction))
    ref_tokens = Counter(tokenize(reference))
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = sum((pred_tokens & ref_tokens).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pred_tokens.values())
    recall = overlap / sum(ref_tokens.values())
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Longest-common-subsequence length, keeping only two DP rows in memory."""
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])
        previous = current
    return previous[-1]


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L: F1 over the longest common *subsequence* of tokens.

    Unlike :func:`token_f1` this is order sensitive, which is what catches an
    answer that reuses the reference's vocabulary while scrambling its claims.
    """
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    """1.0 when the two texts agree once case, punctuation and spacing are gone.

    Two token-empty strings match: that is the abstention case, where expected
    and produced answers are both "nothing".
    """
    return 1.0 if tokenize(prediction) == tokenize(reference) else 0.0


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def aggregate(rows: list[dict]) -> dict[str, float]:
    """Mean and sample standard deviation of every numeric column in ``rows``.

    The mean keeps the original column name and the spread lands under
    ``"<name>_stdev"``, so a report reads ``recall_at_5=0.82`` beside
    ``recall_at_5_stdev=0.31``. ``"n"`` holds the row count and is therefore a
    reserved column name.

    Non-numeric columns (the question, the configuration label) are skipped, as
    are non-finite values, and a column missing from some rows is averaged over
    the rows that do have it. ``bool`` columns - ``abstained``, ``grounded`` -
    average into rates, ``bool`` being an ``int``. Columns are emitted in
    first-seen order rather than set order so that two runs of one sweep produce
    identical reports.
    """
    summary: dict[str, float] = {"n": len(rows)}
    if not rows:
        return summary

    columns: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if key == "n" or not isinstance(value, (int, float)):
                continue
            numeric = float(value)
            if not math.isfinite(numeric):
                continue
            columns.setdefault(key, []).append(numeric)

    for key, values in columns.items():
        summary[key] = statistics.fmean(values)
        summary[f"{key}_stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return summary
