"""Text -> unit vector, with a genuinely useful default that needs no model.

The whole assistant has to run with zero API keys and zero downloads, so the
default embedder here is a signed feature-hashing vectoriser rather than a
neural model. It is not a semantic embedder - it will not match "car" to
"automobile" - but it is a strong lexical vector space (unigrams + bigrams +
character 4-grams, sublinear tf, idf-style prior) that gives the dense leg of
hybrid retrieval something real to contribute, deterministically and in
microseconds. Neural backends slot in behind the same interface when their
optional packages are installed.
"""

from __future__ import annotations

import hashlib
import math
import time
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

from app.config import Settings
from app.errors import ConfigurationError, LLMError
from app.observability import get_logger
from app.utils import normalize_whitespace, tokenize

logger = get_logger(__name__)

__all__ = [
    "Embedder",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "OpenAIEmbedder",
    "get_embedder",
]


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation that leaves all-zero rows alone (no NaNs)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return (matrix / norms).astype(np.float32, copy=False)


def _is_blank(text: str) -> bool:
    return not text or not text.strip()


class Embedder(ABC):
    """Maps text onto a fixed-dimension, L2-normalised float32 vector space.

    Cosine similarity is therefore a plain dot product, which is what every
    vector store backend in this project assumes.
    """

    name: str = "embedder"
    dim: int = 0

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch; returns shape ``(len(texts), dim)`` float32."""

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one query; returns shape ``(dim,)`` float32.

        Symmetric by default. Backends with an asymmetric query encoding (an
        instruction prefix, a separate query tower) override this.
        """
        return self.embed_documents([text])[0]

    def _empty(self, n: int) -> np.ndarray:
        return np.zeros((n, self.dim), dtype=np.float32)


# --------------------------------------------------------------------------- #
# Default backend: signed feature hashing
# --------------------------------------------------------------------------- #
# Family weights stand in for the corpus-wide idf we do not have. Word
# unigrams are the reference channel. Bigrams are discounted because they only
# fire on exact adjacent repetition - useful as a phrase-match bonus, but if
# weighted like unigrams they contribute mostly unmatched mass that dilutes the
# unigram signal after normalisation. Character 4-grams are discounted hardest:
# there are ~5x as many of them per document and neighbouring grams are almost
# perfectly correlated, so their raw mass badly over-counts; at this weight they
# buy morphology ("refund"/"refunds"/"refunded"), typo tolerance and non-space-
# delimited scripts without taking the vector over.
_FAMILY_PRIOR: dict[str, float] = {"word": 1.0, "bigram": 0.7, "char": 0.35}

# Per-feature idf prior. Without corpus statistics the best free signal about a
# term's document frequency is its length: English word frequency falls off
# sharply with length (Zipf), so short tokens are overwhelmingly function words.
# A linear ramp saturating at 8 characters halves the weight of "the"/"of"
# relative to "purchase" while still respecting short content words like "tax".
_PRIOR_SATURATION = 8.0

_CHAR_NGRAM = 4


def _prepare(text: str) -> str:
    """Lowercase, NFKC-fold and collapse all whitespace to single spaces.

    Collapsing newlines matters for the character grams: without it a line break
    inside a sentence produces a different gram sequence than the same sentence
    on one line, and the two would not match.
    """
    return " ".join(normalize_whitespace(text).lower().split())


def _features(text: str) -> Counter[tuple[str, str]]:
    """Count every (family, feature) pair present in ``text``."""
    counts: Counter[tuple[str, str]] = Counter()
    prepared = _prepare(text)
    if not prepared:
        return counts

    words = tokenize(prepared)
    counts.update(("word", word) for word in words)
    counts.update(
        ("bigram", f"{first} {second}") for first, second in zip(words, words[1:], strict=False)
    )
    counts.update(
        ("char", prepared[i : i + _CHAR_NGRAM])
        for i in range(len(prepared) - _CHAR_NGRAM + 1)
    )
    return counts


def _feature_weight(family: str, feature: str, count: int) -> float:
    """Sublinear tf times a static idf-style prior.

    ``1 + log(tf)`` keeps a term repeated twenty times from dominating a term
    seen once by twenty to one; the prior then discounts the feature families
    and the short tokens that carry the least information.
    """
    prior = _FAMILY_PRIOR[family] * min(1.0, (len(feature) + 1) / _PRIOR_SATURATION)
    return (1.0 + math.log(count)) * prior


def _bucket(family: str, feature: str, dim: int) -> tuple[int, float]:
    """Map a feature to a bucket and a sign, stably across processes.

    ``hash()`` is salted per interpreter, so a persisted index would decode
    into noise on the next boot - blake2b is used instead. The sign comes from
    an independent bit of the same digest: it is the standard hashing-trick
    trade (Weinberger et al., 2009) where colliding features cancel in
    expectation instead of piling up, which keeps the inner product an unbiased
    estimate of the un-hashed one.
    """
    digest = hashlib.blake2b(
        f"{family}\x1f{feature}".encode("utf-8", "ignore"), digest_size=8
    ).digest()
    value = int.from_bytes(digest, "big")
    return value % dim, 1.0 if (value >> 63) & 1 else -1.0


class HashingEmbedder(Embedder):
    """Deterministic, dependency-free lexical embedder.

    Same text in, bit-identical vector out - in this process, the next one, and
    on another machine.
    """

    name = "hashing"

    def __init__(self, dim: int = 1024) -> None:
        if dim < 8:
            raise ConfigurationError(
                f"embedding_dim must be at least 8, got {dim}",
                detail="Feature hashing needs room to spread collisions.",
            )
        self.dim = int(dim)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        matrix = self._empty(len(texts))
        for row, text in enumerate(texts):
            for (family, feature), count in _features(text).items():
                index, sign = _bucket(family, feature, self.dim)
                matrix[row, index] += sign * _feature_weight(family, feature, count)
        return _l2_normalise(matrix)


# --------------------------------------------------------------------------- #
# Optional backend: sentence-transformers
# --------------------------------------------------------------------------- #
class SentenceTransformerEmbedder(Embedder):
    """Local neural embedder; requires the ``local-models`` extra."""

    name = "sentence-transformers"

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # ImportError, but also broken installs
            raise ConfigurationError(
                "sentence-transformers is not installed",
                detail="pip install 'rag-assistant[local-models]', "
                "or set RAG_EMBEDDING_BACKEND=hashing",
            ) from exc
        try:
            self._model = SentenceTransformer(model_name)
            self.dim = int(self._model.get_sentence_embedding_dimension())
        except Exception as exc:  # no network on first run, bad model id, ...
            raise ConfigurationError(
                f"could not load sentence-transformers model {model_name!r}",
                detail=str(exc),
            ) from exc
        self.model_name = model_name
        self.batch_size = max(1, batch_size)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        matrix = self._empty(len(texts))
        # Blank strings are dropped rather than encoded: a transformer happily
        # returns a nonzero vector for "", which would make an empty chunk
        # retrievable. The contract says empty text is a zero vector.
        wanted = [i for i, text in enumerate(texts) if not _is_blank(text)]
        if not wanted:
            return matrix
        encoded = self._model.encode(
            [texts[i] for i in wanted],
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        matrix[wanted] = np.asarray(encoded, dtype=np.float32)
        return _l2_normalise(matrix)


# --------------------------------------------------------------------------- #
# Optional backend: any OpenAI-compatible /embeddings endpoint
# --------------------------------------------------------------------------- #
# Dimensions are a property of the model, and the endpoint will not tell us
# before the first call, so known models are tabulated and anything else falls
# back to the configured dim.
_OPENAI_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class OpenAIEmbedder(Embedder):
    """Talks to any OpenAI-compatible ``/embeddings`` endpoint over httpx."""

    name = "openai"

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        batch_size: int = 64,
        timeout_s: float = 60.0,
        max_retries: int = 2,
        dim: int | None = None,
    ) -> None:
        try:
            import httpx
        except Exception as exc:  # pragma: no cover - httpx is a hard dependency
            raise ConfigurationError(
                "httpx is not installed", detail="pip install httpx"
            ) from exc
        if not api_key:
            raise ConfigurationError(
                "no API key configured for the OpenAI embedding backend",
                detail="Set RAG_LLM_API_KEY or use RAG_EMBEDDING_BACKEND=hashing.",
            )
        self._httpx = httpx
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.batch_size = max(1, batch_size)
        self.timeout_s = timeout_s
        self.max_retries = max(0, max_retries)
        self.dim = int(dim) if dim is not None else _OPENAI_DIMS.get(model, 1536)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        matrix = self._empty(len(texts))
        # The API rejects empty strings, and a zero row is the contracted
        # answer for them anyway, so they never leave the process.
        wanted = [i for i, text in enumerate(texts) if not _is_blank(text)]
        if not wanted:
            return matrix
        with self._httpx.Client(timeout=self.timeout_s) as client:
            for start in range(0, len(wanted), self.batch_size):
                batch = wanted[start : start + self.batch_size]
                vectors = self._embed_batch(client, [texts[i] for i in batch])
                for row, vector in zip(batch, vectors, strict=False):
                    matrix[row] = vector
        return _l2_normalise(matrix)

    # ----------------------------------------------------------------- #
    def _embed_batch(self, client: Any, batch: Sequence[str]) -> np.ndarray:
        payload = self._request(
            client, {"model": self.model, "input": list(batch)}
        )
        try:
            rows = sorted(payload["data"], key=lambda item: int(item["index"]))
            vectors = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
        except (KeyError, TypeError, ValueError) as exc:
            raise LLMError(
                "malformed response from the embeddings endpoint", detail=str(exc)
            ) from exc
        if vectors.shape != (len(batch), self.dim):
            raise ConfigurationError(
                f"{self.model} returned {vectors.shape[-1]}-dimensional vectors, "
                f"expected {self.dim}",
                detail="Set RAG_EMBEDDING_DIM to the dimension this model actually returns.",
            )
        return vectors

    def _request(self, client: Any, payload: dict[str, Any]) -> dict[str, Any]:
        """POST with bounded exponential backoff on transient failures."""
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_detail = "unknown error"
        for attempt in range(self.max_retries + 1):
            try:
                response = client.post(url, json=payload, headers=headers)
                if response.status_code < 400:
                    return response.json()
                last_detail = f"HTTP {response.status_code}: {response.text[:200]}"
                if response.status_code not in _RETRYABLE_STATUS:
                    raise LLMError("embeddings request failed", detail=last_detail)
            except self._httpx.HTTPError as exc:
                last_detail = f"{type(exc).__name__}: {exc}"
            if attempt < self.max_retries:
                time.sleep(0.5 * (2**attempt))
        raise LLMError(
            f"embeddings request failed after {self.max_retries + 1} attempts",
            detail=last_detail,
        )


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
_ALIASES = {
    "auto": "auto",
    "hashing": "hashing",
    "hash": "hashing",
    "sentence-transformers": "sentence-transformers",
    "sentencetransformers": "sentence-transformers",
    "sbert": "sentence-transformers",
    "st": "sentence-transformers",
    "local": "sentence-transformers",
    "openai": "openai",
}


def _hashing(settings: Settings) -> HashingEmbedder:
    return HashingEmbedder(dim=settings.embedding_dim)


def _sentence_transformers(settings: Settings) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(
        settings.embedding_model, batch_size=settings.embedding_batch_size
    )


def _openai(settings: Settings) -> OpenAIEmbedder:
    model = settings.openai_embedding_model
    return OpenAIEmbedder(
        model=model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        batch_size=settings.embedding_batch_size,
        timeout_s=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
        # Unknown model behind a custom base_url: trust the operator's configured dim.
        dim=_OPENAI_DIMS.get(model) or settings.embedding_dim,
    )


def get_embedder(settings: Settings) -> Embedder:
    """Build the embedder named by ``settings.embedding_backend``.

    ``auto`` prefers sentence-transformers, then OpenAI when an API key is
    configured, then hashing. A backend that is *named* but unavailable also
    degrades to hashing rather than refusing to start: a missing optional
    package must never take the service down, and the log line says what
    happened. A backend name that is not recognised at all is a config typo and
    does raise, because silently ignoring it would hide the mistake.
    """
    requested = _ALIASES.get(settings.embedding_backend.strip().lower().replace("_", "-"))
    if requested is None:
        raise ConfigurationError(
            f"unknown embedding backend {settings.embedding_backend!r}",
            detail="Expected one of: auto, hashing, sentence-transformers, openai.",
        )

    if requested == "hashing":
        return _hashing(settings)

    if requested == "auto":
        for name, build in (
            ("sentence-transformers", _sentence_transformers),
            ("openai", _openai),
        ):
            if name == "openai" and not settings.llm_api_key:
                continue
            try:
                return build(settings)
            except ConfigurationError as exc:
                logger.info(
                    "embedding backend unavailable",
                    extra={"backend": name, "reason": exc.message},
                )
        return _hashing(settings)

    build = _sentence_transformers if requested == "sentence-transformers" else _openai
    try:
        return build(settings)
    except ConfigurationError as exc:
        logger.warning(
            "requested embedding backend unavailable, falling back to hashing",
            extra={"backend": requested, "reason": exc.message},
        )
        return _hashing(settings)
