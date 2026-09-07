"""Okapi BM25 over an in-memory inverted index.

Pure Python on purpose: the sparse half of hybrid retrieval has to work with no
optional dependency installed, and a corpus that fits in a container is scored
fast enough by a dict of postings. The index is incremental - document
frequencies and the average document length stay exact as documents arrive and
as they are deleted, so a long-lived service never has to rebuild from scratch.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from typing import Any

from app.utils import tokenize


class BM25Index:
    """Incremental BM25 ranking over tokenised documents.

    Scores are raw BM25, not normalised: fusion (`app.retrieval.hybrid`) owns
    the normalisation so that the two retrievers can be compared on equal terms.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        # term -> {doc_id: term frequency}. Document frequency is len(postings),
        # so it can never drift out of sync with the postings themselves.
        self._postings: dict[str, dict[str, int]] = {}
        self._doc_terms: dict[str, tuple[str, ...]] = {}
        self._lengths: dict[str, int] = {}
        self._total_length = 0

    # -- mutation ------------------------------------------------------- #
    def add(self, ids: Iterable[str], texts: Iterable[str]) -> None:
        """Index ``texts`` under ``ids``; re-adding an id replaces its posting.

        ``strict=True`` turns a caller's id/text length mismatch into a loud
        ValueError instead of a silently truncated index.
        """
        for doc_id, text in zip(ids, texts, strict=True):
            if doc_id in self._lengths:
                self._remove(doc_id)
            tokens = tokenize(text)
            self._index(doc_id, Counter(tokens), len(tokens))

    def delete(self, ids: Iterable[str]) -> int:
        """Remove documents, returning how many actually existed."""
        removed = 0
        for doc_id in ids:
            if doc_id in self._lengths:
                self._remove(doc_id)
                removed += 1
        return removed

    def clear(self) -> None:
        self._postings.clear()
        self._doc_terms.clear()
        self._lengths.clear()
        self._total_length = 0

    def _index(self, doc_id: str, freqs: Counter[str], length: int) -> None:
        for term, tf in freqs.items():
            self._postings.setdefault(term, {})[doc_id] = tf
        self._doc_terms[doc_id] = tuple(freqs)
        self._lengths[doc_id] = length
        self._total_length += length

    def _remove(self, doc_id: str) -> None:
        for term in self._doc_terms.pop(doc_id):
            postings = self._postings.get(term)
            if postings is None:
                continue
            postings.pop(doc_id, None)
            if not postings:
                del self._postings[term]
        self._total_length -= self._lengths.pop(doc_id)

    # -- query ---------------------------------------------------------- #
    @property
    def avg_doc_length(self) -> float:
        if not self._lengths:
            return 0.0
        return self._total_length / len(self._lengths)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Top ``k`` ``(doc_id, score)`` pairs, descending, ties broken by id.

        A repeated query term counts once: with queries this short the in-query
        term frequency carries no signal, and collapsing it keeps the iteration
        order of a dict (insertion-ordered, therefore deterministic) rather than
        of a set.
        """
        if k <= 0 or not self._lengths:
            return []
        terms = Counter(tokenize(query))
        if not terms:
            return []

        total_docs = len(self._lengths)
        # A corpus of nothing but empty documents would otherwise divide by zero.
        avgdl = self.avg_doc_length or 1.0
        scores: dict[str, float] = {}
        for term in terms:
            postings = self._postings.get(term)
            if not postings:
                continue
            df = len(postings)
            # Lucene's smoothed idf: strictly positive, so a term that appears
            # in every document cannot push a score negative.
            idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
            for doc_id, tf in postings.items():
                norm = self.k1 * (1.0 - self.b + self.b * self._lengths[doc_id] / avgdl)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf * (self.k1 + 1.0) / (tf + norm)

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:k]

    # -- persistence ---------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable snapshot of the whole index.

        Term frequencies are emitted sorted so two equal indexes serialise to
        byte-identical JSON, which makes index files diffable and hashable.
        """
        documents = [
            {
                "id": doc_id,
                "length": self._lengths[doc_id],
                "freqs": {term: self._postings[term][doc_id] for term in sorted(terms)},
            }
            for doc_id, terms in self._doc_terms.items()
        ]
        return {"version": 1, "k1": self.k1, "b": self.b, "documents": documents}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BM25Index:
        index = cls(k1=float(data.get("k1", 1.5)), b=float(data.get("b", 0.75)))
        for entry in data.get("documents", []):
            freqs: Counter[str] = Counter(
                {str(term): int(tf) for term, tf in entry.get("freqs", {}).items()}
            )
            index._index(str(entry["id"]), freqs, int(entry["length"]))
        return index

    def __len__(self) -> int:
        return len(self._lengths)

    def __contains__(self, doc_id: object) -> bool:
        return doc_id in self._lengths
