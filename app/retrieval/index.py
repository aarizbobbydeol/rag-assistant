"""The retrieval side of the assistant, assembled into one object.

`RagIndex` owns the three things that must stay in lockstep — the chunk table,
the dense vector store and the BM25 index — and exposes a single `search` that
runs dense, lexical or fused retrieval and optionally reranks the result.
Keeping them behind one class is what stops a partially-updated index (vectors
added, BM25 not) from silently degrading recall.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from app.config import Settings
from app.generation.anchors import AnchorIndex, build_anchor_index, chunk_vocab
from app.models import Chunk, Document, RetrievalMode, ScoredChunk
from app.observability import INDEX_DOCS, INDEX_SIZE, get_logger
from app.retrieval.bm25 import BM25Index
from app.retrieval.embeddings import Embedder
from app.retrieval.hybrid import mmr_select, reciprocal_rank_fusion
from app.retrieval.rerank import Reranker
from app.retrieval.vectorstore import VectorStore

logger = get_logger(__name__)

_CHUNKS_FILE = "chunks.json"
_BM25_FILE = "bm25.json"
_VECTORS_FILE = "vectors"

# BM25 score at which the saturating rescale returns 0.5. Chosen from the
# observed range on prose corpora: a genuine keyword match on a short query
# lands around 8-12, an incidental one below 2.
_BM25_HALF_SATURATION = 4.0


class RagIndex:
    """Chunk table + dense store + BM25, kept consistent by construction."""

    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        store: VectorStore,
        reranker: Reranker | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.bm25 = BM25Index(k1=settings.bm25_k1, b=settings.bm25_b)
        self._chunks: dict[str, Chunk] = {}
        self._by_doc: dict[str, list[str]] = {}
        self._anchors: AnchorIndex | None = None
        self._vocabs: dict[str, set[str]] = {}

    # ----------------------------------------------------------------- #
    # Writing
    # ----------------------------------------------------------------- #
    def add_documents(self, docs: Sequence[Document]) -> int:
        from app.ingestion.chunking import get_splitter

        splitter = get_splitter(
            self.settings.splitter,
            self.settings.chunk_size,
            self.settings.chunk_overlap,
            self.settings.min_chunk_tokens,
        )
        chunks: list[Chunk] = []
        for doc in docs:
            self.delete_document(doc.doc_id)
            chunks.extend(splitter.split(doc))
        return self.add_chunks(chunks)

    def add_chunks(self, chunks: Sequence[Chunk]) -> int:
        if not chunks:
            return 0
        ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]

        vectors = self.embedder.embed_documents(texts)
        self.store.add(ids, vectors)
        self.bm25.add(ids, texts)

        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
            bucket = self._by_doc.setdefault(chunk.doc_id, [])
            if chunk.chunk_id not in bucket:
                bucket.append(chunk.chunk_id)

        self._publish_gauges()
        return len(chunks)

    def delete_document(self, doc_id: str) -> int:
        ids = self._by_doc.pop(doc_id, [])
        if not ids:
            return 0
        self.store.delete(ids)
        self.bm25.delete(ids)
        for chunk_id in ids:
            self._chunks.pop(chunk_id, None)
        self._publish_gauges()
        return len(ids)

    def clear(self) -> None:
        self.store.clear()
        self.bm25.clear()
        self._chunks.clear()
        self._by_doc.clear()
        self._publish_gauges()

    # ----------------------------------------------------------------- #
    # Reading
    # ----------------------------------------------------------------- #
    def search(
        self,
        query: str,
        top_k: int,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        rerank: bool = True,
        candidate_k: int | None = None,
    ) -> list[ScoredChunk]:
        if not self._chunks or not query.strip():
            return []

        pool = max(top_k, candidate_k or self.settings.candidate_k)
        dense_hits: list[tuple[str, float]] = []
        lexical_hits: list[tuple[str, float]] = []

        if mode in (RetrievalMode.DENSE, RetrievalMode.HYBRID):
            dense_hits = self.store.search(self.embedder.embed_query(query), pool)
        if mode in (RetrievalMode.LEXICAL, RetrievalMode.HYBRID):
            lexical_hits = self.bm25.search(query, pool)

        dense_scores = dict(dense_hits)
        lexical_scores = dict(lexical_hits)

        if mode is RetrievalMode.HYBRID:
            fused = reciprocal_rank_fusion(
                [dense_hits, lexical_hits],
                k=self.settings.rrf_k,
                weights=[self.settings.dense_weight, self.settings.lexical_weight],
            )
        elif mode is RetrievalMode.DENSE:
            fused = dense_hits
        else:
            fused = lexical_hits
        fused = self._rescale(fused, mode)

        candidates: list[ScoredChunk] = []
        for rank, (chunk_id, score) in enumerate(fused[:pool], start=1):
            chunk = self._chunks.get(chunk_id)
            if chunk is None:  # index and store drifted; skip rather than crash
                continue
            candidates.append(
                ScoredChunk(
                    chunk=chunk,
                    score=float(score),
                    dense_score=dense_scores.get(chunk_id),
                    lexical_score=lexical_scores.get(chunk_id),
                    rank=rank,
                    retriever=mode.value,
                )
            )

        if not candidates:
            return []

        if rerank and self.reranker is not None:
            candidates = self.reranker.rerank(
                query, candidates[: self.settings.rerank_candidates], top_k
            )

        selected = self._diversify(query, candidates, top_k)
        for rank, item in enumerate(selected, start=1):
            item.rank = rank
        return selected

    def _rescale(
        self, hits: list[tuple[str, float]], mode: RetrievalMode
    ) -> list[tuple[str, float]]:
        """Put every mode's score on a comparable 0-1 scale.

        `settings.min_retrieval_score` is a single number used to decide whether
        the corpus knows anything about the question, so the score it is
        compared against has to mean the same thing in all three modes. Raw
        scores do not: cosine sits in [-1, 1], BM25 is unbounded, and an RRF
        score maxes out at ``sum(weights) / (rrf_k + 1)`` - about 0.016 at the
        default settings, which would silently abstain on *every* hybrid query.

        The mapping is monotonic in each mode, so ranking is untouched.
        """
        if not hits:
            return hits
        if mode is RetrievalMode.HYBRID:
            ceiling = (self.settings.dense_weight + self.settings.lexical_weight) / (
                self.settings.rrf_k + 1
            )
            if ceiling <= 0:
                return hits
            return [(doc_id, min(1.0, score / ceiling)) for doc_id, score in hits]
        if mode is RetrievalMode.LEXICAL:
            # Saturating map: BM25 has no upper bound, but the interesting
            # region for "is this relevant at all" is the low end.
            return [(doc_id, score / (score + _BM25_HALF_SATURATION)) for doc_id, score in hits]
        return [(doc_id, max(0.0, score)) for doc_id, score in hits]

    def _diversify(
        self, query: str, candidates: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        """Apply MMR so a single verbose document cannot occupy every slot."""
        if self.settings.mmr_lambda >= 1.0 or len(candidates) <= top_k:
            return candidates[:top_k]
        try:
            vectors = self.embedder.embed_documents([c.chunk.text for c in candidates])
            query_vec = self.embedder.embed_query(query)
            picked = mmr_select(
                query_vec,
                vectors,
                [c.score for c in candidates],
                top_k,
                self.settings.mmr_lambda,
            )
            return [candidates[i] for i in picked]
        except Exception:  # diversity is a nicety, never a reason to fail a query
            logger.warning("mmr_failed_falling_back_to_relevance_order", exc_info=True)
            return candidates[:top_k]

    # ----------------------------------------------------------------- #
    # Persistence
    # ----------------------------------------------------------------- #
    def save(self) -> None:
        if not self.settings.persist_index:
            return
        target = Path(self.settings.index_dir)
        target.mkdir(parents=True, exist_ok=True)
        payload = {
            "embedder": self.embedder.name,
            "dim": self.embedder.dim,
            "chunks": [c.model_dump() for c in self._chunks.values()],
        }
        (target / _CHUNKS_FILE).write_text(json.dumps(payload), encoding="utf-8")
        (target / _BM25_FILE).write_text(json.dumps(self.bm25.to_dict()), encoding="utf-8")
        self.store.save(target / _VECTORS_FILE)
        logger.info("index_saved", extra={"chunks": len(self._chunks), "path": str(target)})

    def load(self) -> bool:
        target = Path(self.settings.index_dir)
        chunks_path = target / _CHUNKS_FILE
        if not chunks_path.exists():
            return False
        try:
            payload = json.loads(chunks_path.read_text(encoding="utf-8"))
            if payload.get("embedder") != self.embedder.name or payload.get("dim") != self.embedder.dim:
                # Vectors written by a different embedder are meaningless here.
                logger.warning(
                    "index_embedder_mismatch_discarding",
                    extra={"stored": payload.get("embedder"), "current": self.embedder.name},
                )
                return False
            chunks = [Chunk(**raw) for raw in payload.get("chunks", [])]
            bm25_path = target / _BM25_FILE
            if bm25_path.exists():
                self.bm25 = BM25Index.from_dict(json.loads(bm25_path.read_text(encoding="utf-8")))
            self.store.load(target / _VECTORS_FILE)
            self._chunks = {c.chunk_id: c for c in chunks}
            self._by_doc = {}
            for chunk in chunks:
                self._by_doc.setdefault(chunk.doc_id, []).append(chunk.chunk_id)
        except Exception:
            logger.warning("index_load_failed_starting_empty", exc_info=True)
            self.clear()
            return False
        self._publish_gauges()
        logger.info("index_loaded", extra={"chunks": len(self._chunks)})
        return True

    # ----------------------------------------------------------------- #
    # Introspection
    # ----------------------------------------------------------------- #
    @property
    def chunks(self) -> dict[str, Chunk]:
        return self._chunks

    @property
    def documents(self) -> dict[str, list[str]]:
        return self._by_doc

    def document_summaries(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for doc_id, chunk_ids in self._by_doc.items():
            if not chunk_ids:
                continue
            first = self._chunks[chunk_ids[0]]
            rows.append(
                {
                    "doc_id": doc_id,
                    "title": first.title,
                    "source": first.source,
                    "chunks": len(chunk_ids),
                    "tokens": sum(self._chunks[c].token_count for c in chunk_ids),
                }
            )
        return sorted(rows, key=lambda r: r["title"])

    def stats(self) -> dict[str, Any]:
        token_counts = [c.token_count for c in self._chunks.values()]
        return {
            "chunks": len(self._chunks),
            "documents": len(self._by_doc),
            "vectors": len(self.store),
            "bm25_documents": len(self.bm25),
            "embedder": self.embedder.name,
            "embedding_dim": self.embedder.dim,
            "vector_store": self.store.name,
            "reranker": self.reranker.name if self.reranker else "none",
            "mean_chunk_tokens": round(float(np.mean(token_counts)), 1) if token_counts else 0.0,
            "max_chunk_tokens": max(token_counts) if token_counts else 0,
        }

    @property
    def anchor_index(self) -> AnchorIndex:
        """Corpus attestation tables, rebuilt lazily after any mutation.

        Every write path already funnels through ``_publish_gauges``, so a single
        invalidation there cannot drift out of step with the chunk table the way
        one assignment per mutation method would.
        """
        if self._anchors is None:
            self._anchors = build_anchor_index([c.text for c in self._chunks.values()])
        return self._anchors

    def chunk_vocabs(self, contexts: Sequence[ScoredChunk]) -> list[set[str]]:
        """Per-chunk stem vocabularies for the retrieved contexts, memoised."""
        vocabs = []
        for scored in contexts:
            cached = self._vocabs.get(scored.chunk.chunk_id)
            if cached is None:
                cached = chunk_vocab(scored.chunk.text)
                self._vocabs[scored.chunk.chunk_id] = cached
            vocabs.append(cached)
        return vocabs

    def _publish_gauges(self) -> None:
        self._anchors = None
        self._vocabs = {}
        INDEX_SIZE.set(len(self._chunks))
        INDEX_DOCS.set(len(self._by_doc))

    def __len__(self) -> int:
        return len(self._chunks)
