"""Core data contracts shared by every layer of the RAG assistant.

Everything that crosses a module boundary is defined here so that the
ingestion, retrieval, generation and evaluation packages can be developed
and tested independently.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Corpus primitives
# --------------------------------------------------------------------------- #
class Document(BaseModel):
    """A whole source file after text extraction, before chunking."""

    doc_id: str
    source: str
    title: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A retrievable unit of text with enough provenance to cite it."""

    chunk_id: str
    doc_id: str
    source: str
    title: str
    text: str
    ordinal: int = 0
    start_char: int = 0
    end_char: int = 0
    page: int | None = None
    token_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def locator(self) -> str:
        """Human readable pointer used in citation footers."""
        if self.page is not None:
            return f"{self.title} (p.{self.page})"
        return f"{self.title} (chunk {self.ordinal})"


class ScoredChunk(BaseModel):
    """A chunk plus every score that contributed to its ranking."""

    chunk: Chunk
    score: float = 0.0
    dense_score: float | None = None
    lexical_score: float | None = None
    rerank_score: float | None = None
    rank: int = 0
    retriever: str = "hybrid"


# --------------------------------------------------------------------------- #
# Answer primitives
# --------------------------------------------------------------------------- #
class Citation(BaseModel):
    """Resolved `[n]` marker -> the chunk that backs it."""

    marker: int
    chunk_id: str
    doc_id: str
    source: str
    title: str
    page: int | None = None
    quote: str = ""
    score: float = 0.0


class SentenceSupport(BaseModel):
    """Per-sentence groundedness verdict from the hallucination guard."""

    sentence: str
    supported: bool
    support_score: float
    cited_markers: list[int] = Field(default_factory=list)


class Groundedness(BaseModel):
    """Aggregate verdict on whether the answer is backed by the context."""

    score: float = 0.0
    supported_sentences: int = 0
    total_sentences: int = 0
    sentences: list[SentenceSupport] = Field(default_factory=list)
    unsupported: list[str] = Field(default_factory=list)
    invalid_citations: list[int] = Field(default_factory=list)
    uncited_sentences: int = 0
    abstained: bool = False
    reason: str | None = None


class Usage(BaseModel):
    """Token / cost accounting for one generation call."""

    provider: str = "none"
    model: str = "none"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0

    def merged(self, other: Usage) -> Usage:
        return Usage(
            provider=other.provider or self.provider,
            model=other.model or self.model,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            estimated_cost_usd=self.estimated_cost_usd + other.estimated_cost_usd,
        )


class AnswerResult(BaseModel):
    """Everything the pipeline knows about one answered question."""

    question: str
    standalone_question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    contexts: list[ScoredChunk] = Field(default_factory=list)
    groundedness: Groundedness = Field(default_factory=Groundedness)
    usage: Usage = Field(default_factory=Usage)
    latency_ms: dict[str, float] = Field(default_factory=dict)
    session_id: str | None = None
    trace_id: str = ""


# --------------------------------------------------------------------------- #
# Conversation
# --------------------------------------------------------------------------- #
class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    trace_id: str | None = None


class LLMMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMResponse(BaseModel):
    text: str
    usage: Usage = Field(default_factory=Usage)
    finish_reason: str = "stop"


# --------------------------------------------------------------------------- #
# Enumerations used by config and the API
# --------------------------------------------------------------------------- #
class RetrievalMode(str, Enum):
    DENSE = "dense"
    LEXICAL = "lexical"
    HYBRID = "hybrid"


class SplitterName(str, Enum):
    RECURSIVE = "recursive"
    FIXED = "fixed"
    SENTENCE = "sentence"
    SEMANTIC = "semantic"


# --------------------------------------------------------------------------- #
# HTTP request / response models
# --------------------------------------------------------------------------- #
class IngestPathsRequest(BaseModel):
    paths: list[str] = Field(..., description="Files or directories to ingest.")
    recursive: bool = True
    replace: bool = Field(False, description="Drop the existing index first.")


class IngestedDoc(BaseModel):
    doc_id: str
    source: str
    title: str
    chunks: int
    characters: int


class IngestResponse(BaseModel):
    documents: list[IngestedDoc]
    total_documents: int
    total_chunks: int
    skipped: list[str] = Field(default_factory=list)
    elapsed_ms: float = 0.0
    index_size: int = 0


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str | None = None
    top_k: int | None = Field(None, ge=1, le=50)
    mode: RetrievalMode | None = None
    rerank: bool | None = None
    strict: bool | None = Field(
        None, description="Abstain instead of returning an ungrounded answer."
    )
    include_contexts: bool = True


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)
    mode: RetrievalMode | None = None
    rerank: bool | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[ScoredChunk]
    elapsed_ms: float = 0.0


class HealthResponse(BaseModel):
    status: str
    version: str
    index_size: int
    documents: int
    embedder: str
    llm_provider: str
    llm_model: str
    llm_available: bool
    vector_store: str
    reranker: str


class SessionResponse(BaseModel):
    session_id: str
    turns: list[Turn]


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    trace_id: str | None = None
