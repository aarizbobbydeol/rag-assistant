"""The orchestration layer: ingest documents, answer questions about them.

`RagPipeline` is the only place that knows the full sequence — condense,
retrieve, gate, generate, verify, remember — and it is deliberately the only
stateful object in the process.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

from app.config import Settings
from app.errors import DocumentLoadError, UnsupportedFileType
from app.generation.anchors import AnchorMiss
from app.generation.citations import renumber_answer
from app.generation.guardrails import (
    abstention_message,
    anchor_support,
    check_groundedness,
    money_slot_unfilled,
    query_term_coverage,
    should_abstain,
)
from app.generation.llm import LLMClient, get_llm
from app.generation.prompts import build_answer_messages, build_condense_messages
from app.ingestion.loaders import iter_paths, load_bytes, load_path
from app.memory import SessionStore
from app.models import (
    AnswerResult,
    Document,
    Groundedness,
    IngestedDoc,
    IngestResponse,
    RetrievalMode,
    ScoredChunk,
    Turn,
    Usage,
)
from app.observability import (
    ABSTENTIONS,
    GROUNDEDNESS,
    LLM_COST,
    LLM_ERRORS,
    LLM_TOKENS,
    QUESTIONS,
    RETRIEVED_TOP_SCORE,
    get_logger,
    timed,
)
from app.retrieval.embeddings import get_embedder
from app.retrieval.index import RagIndex
from app.retrieval.rerank import get_reranker
from app.retrieval.vectorstore import get_vector_store
from app.utils import content_words, new_trace_id, tokenize, truncate

logger = get_logger(__name__)

# A question opening with one of these is continuing the previous one rather
# than starting a new topic. Interrogatives are deliberately absent: "what",
# "which" and "how" open most *self-contained* questions too.
_CONTINUATION_OPENERS = frozenset({"and", "also", "but", "so", "then", "plus"})
_CONTINUATION_PHRASES = ("what about", "how about", "and what about", "what if")

# Below this many content words a question cannot be retrieved against on its
# own - "are there any exceptions?" carries exactly one.
_SELF_CONTAINED_MIN_TERMS = 3


def _needs_context(question: str) -> bool:
    """Whether a follow-up has to borrow terms from the previous question.

    Two independent signals, either being enough: the question opens as a
    continuation ("and what about..."), or it is too thin to stand alone. The
    test has to stay conservative in the other direction too - prepending the
    previous question to a self-contained one drags in its search terms and
    buries the real answer.
    """
    tokens = tokenize(question)
    if not tokens:
        return False
    if tokens[0] in _CONTINUATION_OPENERS:
        return True
    lowered = question.strip().lower()
    if any(lowered.startswith(phrase) for phrase in _CONTINUATION_PHRASES):
        return True
    return len(content_words(question)) < _SELF_CONTAINED_MIN_TERMS


class RagPipeline:
    """Everything the service can do, minus the HTTP shell."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        settings.ensure_dirs()

        self.embedder = get_embedder(settings)
        self.store = get_vector_store(settings, self.embedder.dim)
        self.llm: LLMClient = get_llm(settings)
        self.reranker = get_reranker(settings, self.llm)
        self.index = RagIndex(settings, self.embedder, self.store, self.reranker)
        self.sessions = SessionStore(
            max_sessions=settings.max_sessions, ttl_s=settings.session_ttl_s
        )

        if settings.persist_index and self.index.load():
            logger.info("index_restored", extra={"chunks": len(self.index)})

        logger.info(
            "pipeline_ready",
            extra={
                "embedder": self.embedder.name,
                "vector_store": self.store.name,
                "reranker": self.reranker.name,
                "llm": f"{self.llm.provider}:{self.llm.model}",
                "llm_available": self.llm.available,
            },
        )

    # ----------------------------------------------------------------- #
    # Ingestion
    # ----------------------------------------------------------------- #
    def ingest_paths(
        self, paths: Sequence[str | Path], recursive: bool = True, replace: bool = False
    ) -> IngestResponse:
        started = time.perf_counter()
        if replace:
            self.index.clear()

        targets: list[Path] = []
        skipped: list[str] = []
        for raw in paths:
            path = Path(raw).expanduser()
            if path.is_dir():
                targets.extend(iter_paths(path, recursive=recursive))
            elif path.is_file():
                targets.append(path)
            else:
                skipped.append(f"{path}: not found")

        docs: list[Document] = []
        for target in targets:
            try:
                docs.append(load_path(target))
            except (UnsupportedFileType, DocumentLoadError) as exc:
                skipped.append(f"{target.name}: {exc.message}")

        return self._index_documents(docs, skipped, started)

    def ingest_uploads(
        self, files: Sequence[tuple[str, bytes]], replace: bool = False
    ) -> IngestResponse:
        started = time.perf_counter()
        if replace:
            self.index.clear()

        docs: list[Document] = []
        skipped: list[str] = []
        upload_dir = Path(self.settings.upload_dir)
        for filename, payload in files:
            try:
                doc = load_bytes(payload, filename)
            except (UnsupportedFileType, DocumentLoadError) as exc:
                skipped.append(f"{filename}: {exc.message}")
                continue
            # Keep the original bytes so an index rebuild does not need the client.
            (upload_dir / Path(filename).name).write_bytes(payload)
            docs.append(doc)

        return self._index_documents(docs, skipped, started)

    def _index_documents(
        self, docs: list[Document], skipped: list[str], started: float
    ) -> IngestResponse:
        rows: list[IngestedDoc] = []
        total_chunks = 0
        for doc in docs:
            added = self.index.add_documents([doc])
            total_chunks += added
            rows.append(
                IngestedDoc(
                    doc_id=doc.doc_id,
                    source=doc.source,
                    title=doc.title,
                    chunks=added,
                    characters=len(doc.text),
                )
            )

        if docs:
            self.index.save()

        elapsed = (time.perf_counter() - started) * 1000
        logger.info(
            "ingest_complete",
            extra={
                "documents": len(rows),
                "chunks": total_chunks,
                "skipped": len(skipped),
                "elapsed_ms": round(elapsed, 1),
            },
        )
        return IngestResponse(
            documents=rows,
            total_documents=len(rows),
            total_chunks=total_chunks,
            skipped=skipped,
            elapsed_ms=round(elapsed, 2),
            index_size=len(self.index),
        )

    # ----------------------------------------------------------------- #
    # Question answering
    # ----------------------------------------------------------------- #
    def answer(
        self,
        question: str,
        session_id: str | None = None,
        top_k: int | None = None,
        mode: RetrievalMode | None = None,
        rerank: bool | None = None,
        strict: bool | None = None,
        include_contexts: bool = True,
    ) -> AnswerResult:
        cfg = self.settings
        top_k = top_k or cfg.top_k
        mode = mode or cfg.retrieval_mode
        rerank = cfg.rerank_enabled if rerank is None else rerank
        strict = cfg.strict_grounding if strict is None else strict
        trace_id = new_trace_id()
        latency: dict[str, float] = {}
        usage = Usage()

        history = self.sessions.history(session_id, cfg.history_turns)

        with timed(latency, "condense"):
            standalone = self._condense(question, history)

        with timed(latency, "retrieve"):
            contexts = self.index.search(standalone, top_k, mode, rerank)

        top_score = contexts[0].score if contexts else 0.0
        RETRIEVED_TOP_SCORE.observe(max(0.0, min(1.0, top_score)))

        # Two independent pre-generation gates. The score gate catches "nothing
        # in the corpus is close"; the coverage gate catches the subtler case
        # where retrieval returned something plausible but the question's actual
        # subject appears nowhere in it - the answer would then be grounded in a
        # passage that does not address the question.
        coverage = query_term_coverage(standalone, contexts)
        reason = None
        anchors: list[AnchorMiss] = []
        if not contexts:
            reason = "empty_index" if not len(self.index) else "no_results"
        elif should_abstain(top_score, cfg):
            reason = "low_retrieval_score"
        elif coverage < cfg.min_query_coverage:
            reason = "query_terms_absent"
        elif anchors := anchor_support(
            standalone, self.index.chunk_vocabs(contexts), self.index.anchor_index, cfg
        ):
            # Coverage is a bag of unigrams pooled over every retrieved chunk, so
            # it reads 1.0 on "the SEV-4 acknowledgement target" - the corpus has
            # "SEV", it has digits, and it even has the string SEV-4 inside the
            # sentence saying no such severity exists. This gate asks the sharper
            # question: is each discriminative term the question names actually
            # attested, unnegated, in the passages we are about to answer from?
            reason = "anchor_missing"
            logger.info(
                "anchor_miss",
                extra={"trace_id": trace_id, "anchors": [str(a) for a in anchors]},
            )

        if reason is not None:
            return self._abstain(
                question, standalone, contexts, session_id, trace_id, latency, usage, reason,
                include_contexts, anchors,
            )

        with timed(latency, "generate"):
            messages = build_answer_messages(
                standalone, contexts, history, cfg.max_context_tokens
            )
            try:
                completion = self.llm.complete(
                    messages, max_tokens=cfg.llm_max_tokens, temperature=cfg.llm_temperature
                )
            except Exception as exc:
                LLM_ERRORS.labels(kind=type(exc).__name__).inc()
                logger.error("llm_call_failed", extra={"error": str(exc)}, exc_info=True)
                raise

        usage = usage.merged(completion.usage)
        self._record_usage(completion.usage)

        with timed(latency, "verify"):
            answer_text, citations, invalid = renumber_answer(completion.text, contexts)
            grounded = check_groundedness(
                answer_text, contexts, citations, invalid, cfg, self.embedder
            )

        if money_slot_unfilled(standalone, answer_text, cfg):
            # The tier table is full of numbers, so a price question retrieves it
            # happily and the answer quotes requests-per-minute as though that
            # were a price. Every clause is grounded; none of it is money.
            ABSTENTIONS.labels(reason="money_slot_unfilled").inc()
            answer_text = abstention_message("money_slot_unfilled", contexts, (), cfg)
            citations = []
            grounded = Groundedness(
                score=1.0, abstained=True, reason="money_slot_unfilled"
            )

        if strict and grounded.abstained:
            ABSTENTIONS.labels(reason="ungrounded").inc()
            answer_text = cfg.abstain_message
            citations = []

        GROUNDEDNESS.observe(grounded.score)
        QUESTIONS.labels(outcome="abstained" if grounded.abstained else "answered").inc()

        if session_id:
            self.sessions.append(session_id, Turn(role="user", content=question))
            self.sessions.append(
                session_id, Turn(role="assistant", content=answer_text, trace_id=trace_id)
            )

        logger.info(
            "question_answered",
            extra={
                "trace_id": trace_id,
                "contexts": len(contexts),
                "citations": len(citations),
                "groundedness": grounded.score,
                "top_score": round(top_score, 4),
                "tokens": usage.total_tokens,
                "cost_usd": round(usage.estimated_cost_usd, 6),
                "latency_ms": latency,
            },
        )

        return AnswerResult(
            question=question,
            standalone_question=standalone,
            answer=answer_text,
            citations=citations,
            contexts=contexts if include_contexts else [],
            groundedness=grounded,
            usage=usage,
            latency_ms=latency,
            session_id=session_id,
            trace_id=trace_id,
        )

    # ----------------------------------------------------------------- #
    # Helpers
    # ----------------------------------------------------------------- #
    def _condense(self, question: str, history: Sequence[Turn]) -> str:
        """Rewrite a follow-up into a standalone question.

        Without this, "and how long does that take?" retrieves nothing useful:
        the pronoun carries all the meaning and none of the search terms.
        """
        if not history or not self.settings.condense_questions:
            return question
        if not (self.llm.available and self.llm.generative):
            # An extractive backend can only quote the context back, so asking
            # it to rewrite a question yields an abstention that would then be
            # used as the search query. Fall back deterministically instead -
            # but only for a question that actually needs the context, because
            # prepending the previous question to a self-contained one drags in
            # its search terms and buries the real answer.
            if not _needs_context(question):
                return question
            previous = next(
                (t.content for t in reversed(history) if t.role == "user"), ""
            )
            return f"{previous} {question}".strip() if previous else question
        try:
            response = self.llm.complete(
                build_condense_messages(question, history), max_tokens=120, temperature=0.0
            )
            self._record_usage(response.usage)
            rewritten = response.text.strip().strip('"')
            return rewritten or question
        except Exception:
            logger.warning("condense_failed_using_raw_question", exc_info=True)
            return question

    def _abstain(
        self,
        question: str,
        standalone: str,
        contexts: list[ScoredChunk],
        session_id: str | None,
        trace_id: str,
        latency: dict[str, float],
        usage: Usage,
        reason: str,
        include_contexts: bool,
        anchors: Sequence[AnchorMiss] = (),
    ) -> AnswerResult:
        ABSTENTIONS.labels(reason=reason).inc()
        QUESTIONS.labels(outcome="abstained").inc()
        message = abstention_message(reason, contexts, anchors, self.settings)
        if session_id:
            self.sessions.append(session_id, Turn(role="user", content=question))
            self.sessions.append(
                session_id, Turn(role="assistant", content=message, trace_id=trace_id)
            )
        logger.info(
            "question_abstained",
            extra={"trace_id": trace_id, "reason": reason, "question": truncate(question, 120)},
        )
        return AnswerResult(
            question=question,
            standalone_question=standalone,
            answer=message,
            citations=[],
            contexts=contexts if include_contexts else [],
            groundedness=Groundedness(score=1.0, abstained=True, reason=reason),
            usage=usage,
            latency_ms=latency,
            session_id=session_id,
            trace_id=trace_id,
        )

    def _record_usage(self, usage: Usage) -> None:
        if usage.prompt_tokens:
            LLM_TOKENS.labels(kind="prompt", model=usage.model).inc(usage.prompt_tokens)
        if usage.completion_tokens:
            LLM_TOKENS.labels(kind="completion", model=usage.model).inc(usage.completion_tokens)
        if usage.estimated_cost_usd:
            LLM_COST.labels(model=usage.model).inc(usage.estimated_cost_usd)

    def health(self) -> dict[str, object]:
        stats = self.index.stats()
        return {
            "status": "ok",
            "version": self.settings.version,
            "index_size": stats["chunks"],
            "documents": stats["documents"],
            "embedder": self.embedder.name,
            "llm_provider": self.llm.provider,
            "llm_model": self.llm.model,
            "llm_available": self.llm.available,
            "vector_store": self.store.name,
            "reranker": self.reranker.name,
        }
