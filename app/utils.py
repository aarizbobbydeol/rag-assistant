"""Small helpers shared across modules (tokens, ids, text hygiene)."""

from __future__ import annotations

import hashlib
import re
import unicodedata
import uuid

_WORD_RE = re.compile(r"[A-Za-z0-9_]+(?:'[A-Za-z]+)?")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_MULTI_NL = re.compile(r"\n{3,}")

# Empirically ~0.75 words per token for English prose; good enough for budget
# accounting and chunk sizing without pulling in a tokenizer dependency.
_TOKENS_PER_WORD = 1.32

_ENCODER_CACHE: list[object] = []


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


def stable_id(*parts: str) -> str:
    digest = hashlib.sha1("\x1f".join(parts).encode("utf-8", "ignore")).hexdigest()
    return digest[:16]


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; used by BM25, overlap scoring and eval metrics."""
    return _WORD_RE.findall(text.lower())


def _get_tiktoken_encoder():  # pragma: no cover - depends on an optional package
    if _ENCODER_CACHE:
        return _ENCODER_CACHE[0]
    try:
        import tiktoken

        encoder = tiktoken.get_encoding("cl100k_base")
    except Exception:
        encoder = None
    _ENCODER_CACHE.append(encoder)
    return encoder


def estimate_tokens(text: str) -> int:
    """Cheap deterministic token estimate.

    Uses ``tiktoken`` when it happens to be installed so cost reporting is exact
    against OpenAI models, and falls back to a word-count heuristic otherwise.
    """
    if not text:
        return 0
    encoder = _get_tiktoken_encoder()
    if encoder is not None:
        return len(encoder.encode(text))
    words = len(_WORD_RE.findall(text))
    return max(1, int(round(words * _TOKENS_PER_WORD)))


# Ordered longest-first so that "ations" wins over "ions" over "s". Each entry
# is (suffix, replacement); the replacement keeps related forms colliding
# ("severities" -> "severity") instead of merely truncating.
_STEM_RULES: tuple[tuple[str, str], ...] = (
    ("ational", "ate"),
    ("ization", "ize"),
    ("iveness", "ive"),
    ("ations", "ate"),
    ("ements", "ement"),
    ("ities", "ity"),
    ("ation", "ate"),
    ("ings", ""),
    ("ies", "y"),
    ("ied", "y"),
    ("ing", ""),
    ("ers", ""),
    ("ly", ""),
    ("er", ""),
    ("ed", ""),
    ("es", ""),
    ("s", ""),
)
_MIN_STEM_LENGTH = 3


def stem(word: str) -> str:
    """A deliberately crude suffix stripper for lexical overlap scoring.

    It is not linguistically correct and is not meant to be. Its whole job is to
    make ordinary inflections collide - ``paging``/``page``,
    ``requires``/``required`` - so that overlap scoring survives the paraphrase
    between how a question is asked and how a document is written.

    It is conservative on short words because over-stemming invents matches, and
    a false match is worse than a missed one anywhere a citation is being
    checked. Applying one rule and then a single trailing ``e`` strip keeps it
    idempotent-in-effect: ``page`` and ``paging`` both land on ``pag``.
    """
    lowered = word.lower()
    if len(lowered) <= _MIN_STEM_LENGTH:
        return lowered

    for suffix, replacement in _STEM_RULES:
        if lowered.endswith(suffix):
            base = lowered[: -len(suffix)] + replacement
            if len(base) >= _MIN_STEM_LENGTH:
                lowered = base
                break

    if len(lowered) > _MIN_STEM_LENGTH and lowered.endswith("e"):
        lowered = lowered[:-1]
    return lowered


def stem_tokens(text: str) -> list[str]:
    """``tokenize`` followed by :func:`stem`, preserving order and duplicates."""
    return [stem(token) for token in tokenize(text)]


def split_sentences(text: str) -> list[str]:
    """Regex sentence splitter that keeps list items and headings intact."""
    if not text:
        return []
    pieces: list[str] = []
    for block in text.split("\n"):
        block = block.strip()
        if not block:
            continue
        pieces.extend(part.strip() for part in _SENTENCE_END.split(block) if part.strip())
    return pieces


# A general-purpose English stopword list. `app.generation.citations` and
# `app.generation.llm` keep their own tuned variants deliberately; this one is
# for callers that just need "does this text carry any topical content".
STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are aren as at be because been
    before being below between both but by can cannot could couldn did didn do does doesn
    doing don down during each few for from further had hadn has hasn have haven having he
    her here hers herself him himself his how i if in into is isn it its itself just me more
    most must my myself no nor not now of off on once only or other ought our ours ourselves
    out over own same shan she should shouldn so some such than that the their theirs them
    themselves then there these they this those through to too under until up very was wasn
    we were weren what when where which while who whom why will with won would wouldn you
    your yours yourself yourselves
    """.split()
)


def content_words(text: str) -> list[str]:
    """Topical tokens: no stopwords, nothing shorter than three characters."""
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 2]


def normalize_whitespace(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "")
    text = _WS_RE.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def truncate(text: str, limit: int, suffix: str = "...") -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(suffix))].rstrip() + suffix


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def containment(needle: set[str], haystack: set[str]) -> float:
    """Fraction of ``needle`` covered by ``haystack`` - asymmetric on purpose."""
    if not needle:
        return 0.0
    return len(needle & haystack) / len(needle)
