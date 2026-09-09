"""Single source of truth for runtime configuration.

Every knob is settable through the environment (prefix ``RAG_``) or a local
``.env`` file, which keeps the container image identical across deployments.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.models import RetrievalMode, SplitterName

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=(".env",),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- service -------------------------------------------------------- #
    app_name: str = "rag-assistant"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    log_json: bool = True
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # -- storage -------------------------------------------------------- #
    data_dir: Path = PROJECT_ROOT / "data"
    index_dir: Path = PROJECT_ROOT / "data" / "index"
    corpus_dir: Path = PROJECT_ROOT / "data" / "corpus"
    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"
    persist_index: bool = True

    # -- chunking ------------------------------------------------------- #
    splitter: SplitterName = SplitterName.RECURSIVE
    chunk_size: int = Field(700, ge=64, le=8000, description="Target chunk size in tokens.")
    chunk_overlap: int = Field(120, ge=0, le=4000)
    min_chunk_tokens: int = 24

    # -- embeddings ----------------------------------------------------- #
    embedding_backend: str = Field(
        "auto", description="auto | hashing | sentence-transformers | openai"
    )
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = Field(1024, description="Dimension for the built-in hashing embedder.")
    embedding_batch_size: int = 64

    # -- vector store --------------------------------------------------- #
    vector_store: str = Field(
        "auto", description="auto | numpy | faiss | pgvector | qdrant"
    )
    pg_dsn: str = "postgresql://rag:rag@localhost:5432/rag"
    pg_table: str = "rag_chunks"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "rag_chunks"

    # -- auth ----------------------------------------------------------- #
    auth_required: bool = Field(
        False, description="Require a bearer token on every /api route."
    )
    auth_db_path: Path = PROJECT_ROOT / "data" / "users.sqlite3"
    auth_secret: str = Field(
        "dev-insecure-change-me",
        description="HS256 signing key. MUST be overridden in production.",
    )
    auth_token_ttl_s: int = Field(60 * 60 * 12, ge=60)
    auth_allow_registration: bool = True
    auth_min_password_length: int = Field(8, ge=6)

    # -- retrieval ------------------------------------------------------ #
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    top_k: int = Field(5, ge=1, le=50, description="Contexts handed to the LLM.")
    candidate_k: int = Field(24, ge=1, le=200, description="Candidates fetched per retriever.")
    rrf_k: int = Field(60, ge=1, description="Reciprocal-rank-fusion damping constant.")
    dense_weight: float = 0.5
    lexical_weight: float = 0.5
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    mmr_lambda: float = Field(0.7, ge=0.0, le=1.0, description="1.0 = pure relevance.")

    # -- reranking ------------------------------------------------------ #
    rerank_enabled: bool = True
    rerank_backend: str = Field("auto", description="auto | heuristic | cross-encoder | llm | none")
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = Field(24, ge=1, le=200)

    # -- generation ----------------------------------------------------- #
    llm_backend: str = Field("auto", description="auto | openai | extractive")
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_temperature: float = 0.0
    llm_max_tokens: int = 800
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2
    llm_price_input_per_1m: float = 0.15
    llm_price_output_per_1m: float = 0.60
    max_context_tokens: int = 6000

    # -- conversation --------------------------------------------------- #
    history_turns: int = Field(6, ge=0, le=50)
    condense_questions: bool = True
    session_ttl_s: int = 21600
    max_sessions: int = 1000

    # -- safety --------------------------------------------------------- #
    strict_grounding: bool = Field(
        False, description="Replace ungrounded answers with an explicit abstention."
    )
    min_retrieval_score: float = Field(
        0.10, description="Below this top-1 score the assistant refuses to answer."
    )
    min_query_coverage: float = Field(
        0.70,
        description=(
            "Fraction of the question's subject terms that must appear somewhere in "
            "the retrieved context before the assistant will answer at all. "
            "Interrogative scaffolding is excluded from the denominator, so this "
            "sits higher than a raw content-word fraction would."
        ),
    )
    anchor_guard_enabled: bool = Field(
        True,
        description=(
            "Refuse when a discriminative term the question names - an identifier "
            "like SEV-4, a multi-word name, or an adjacent pair of modifiers - is "
            "never positively attested in the retrieved documents."
        ),
    )
    anchor_detectors: list[str] = Field(
        default_factory=lambda: ["identifier", "name_phrase", "orphan_span", "code_identifier"],
        description="Which anchor detectors are active. Dropping one only loosens the guard.",
    )
    anchor_min_orphan_span: int = Field(
        2,
        ge=2,
        le=6,
        description=(
            "How many adjacent question terms must be absent from the best-matching "
            "chunk before that counts as evidence the passage is off-target."
        ),
    )
    money_slot_guard: bool = Field(
        True,
        description="Refuse when a question asks a price and the answer names no money.",
    )
    groundedness_threshold: float = Field(
        0.42, description="Per-sentence support score required to count as grounded."
    )
    min_answer_groundedness: float = Field(
        0.60, description="Fraction of sentences that must be supported in strict mode."
    )
    abstain_message: str = "I could not find enough support for that in the indexed documents."

    # -- privacy -------------------------------------------------------- #
    # Defaults chosen so that running this service stores nothing about the
    # people using it. Both can be turned on deliberately; neither turns itself
    # on because a convenience needed it.
    store_uploads: bool = Field(
        False,
        description=(
            "Write uploaded file bytes to disk. Off by default: the text is "
            "already in the index, and keeping the original means holding "
            "someone's document until an operator deletes it. Turning this on "
            "lets the index be rebuilt without the client re-uploading."
        ),
    )
    log_question_text: bool = Field(
        False,
        description=(
            "Include the question itself in logs. Off by default, because a "
            "question is user content and logs travel further than the service "
            "does. Trace ids still tie a request together without it."
        ),
    )

    # -- ops ------------------------------------------------------------ #
    metrics_enabled: bool = True
    max_upload_mb: int = 32

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.index_dir, self.corpus_dir, self.upload_dir):
            Path(path).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Used by tests that mutate the environment."""
    get_settings.cache_clear()
