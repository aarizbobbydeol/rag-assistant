"""Resolution of ``[n]`` citation markers against the retrieved contexts.

The prompt renders contexts as a 1-based numbered list, so a marker is nothing
but a position in that list. This module turns those positions back into
:class:`~app.models.Citation` records, quarantines the ones the model
hallucinated, and rewrites the prose so the surviving markers form a dense
``1..n`` sequence that lines up with the returned citation list.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.models import Citation, ScoredChunk
from app.utils import (
    containment,
    jaccard,
    normalize_whitespace,
    split_sentences,
    tokenize,
    truncate,
)

# A marker group is a bracket containing nothing but digits and commas, so
# ``[not a number]``, ``[]`` and the ``[i]`` of ordinary prose never match.
_MARKER_GROUP_RE = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")

# Models do not reliably emit ASCII brackets. Several - gpt-oss among them -
# cite with CJK lenticular brackets, and some prompt styles come back with
# fullwidth or mathematical variants. The citation is perfectly well formed in
# every case, so folding the variants onto ASCII before parsing is the
# difference between reading the model's citations and scoring it as having
# cited nothing at all.
_BRACKET_VARIANTS = str.maketrans(
    {
        "【": "[",  # 【 left lenticular
        "】": "]",  # 】 right lenticular
        "［": "[",  # ［ fullwidth
        "］": "]",  # ］ fullwidth
        "⟦": "[",  # ⟦ mathematical
        "⟧": "]",  # ⟧ mathematical
        "❨": "[",  # ❨ ornate
        "❩": "]",
    }
)


def normalize_markers(text: str) -> str:
    """Fold unicode bracket variants onto ASCII so markers are parseable."""
    return text.translate(_BRACKET_VARIANTS)


_SPACE_BEFORE_PUNCT_RE = re.compile(r"[ \t]+([.,;:!?)\]}])")
# Requires a preceding non-space so line indentation survives the cleanup.
_INTERIOR_RUN_RE = re.compile(r"(?<=\S)[ \t]{2,}")
_TRAILING_SPACE_RE = re.compile(r"[ \t]+(?=\n)")

QUOTE_MAX_CHARS = 240

# Trimmed so the overlap score reflects topic words rather than grammar. Kept
# local: `app.utils.tokenize` is deliberately stopword-agnostic because BM25
# wants the full term stream.
_STOPWORDS = frozenset(
    ["a", "about", "after", "all", "also", "an", "and", "any", "are", "as", "at", "be", "been", "before", "being", "but", "by", "can", "could", "did", "do", "does", "for", "from", "had", "has", "have", "he", "her", "his", "how", "i", "if", "in", "into", "is", "it", "its", "may", "might", "more", "most", "must", "no", "not", "of", "on", "only", "or", "other", "our", "over", "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "then", "there", "these", "they", "this", "those", "to", "too", "us", "very", "was", "we", "were", "what", "when", "where", "which", "who", "why", "will", "with", "would", "you", "your"]
)


def extract_markers(text: str) -> list[int]:
    """Marker numbers in order of first appearance, deduplicated.

    Handles ``[1]``, ``[1,2]``, ``[1, 2]`` and ``[1][2]`` alike; a dict is used
    as the ordered set so the result is deterministic across processes.
    """
    seen: dict[int, None] = {}
    for match in _MARKER_GROUP_RE.finditer(normalize_markers(text)):
        for part in match.group(1).split(","):
            seen.setdefault(int(part.strip()), None)
    return list(seen)


def resolve_citations(
    answer: str, contexts: Sequence[ScoredChunk]
) -> tuple[list[Citation], list[int]]:
    """Map 1-based markers onto ``contexts`` positionally.

    Returns ``(citations, invalid_markers)``, both ordered by first appearance
    in ``answer``. A marker is invalid when it is zero, negative or points past
    the end of the context list - i.e. the model invented a source.
    """
    focus = _focus_sentences(answer)
    citations: list[Citation] = []
    invalid: list[int] = []
    for marker in extract_markers(answer):
        # `<= 0` is unreachable through `extract_markers` (the pattern has no
        # sign) but keeps the range check honest if the pattern ever widens.
        if marker <= 0 or marker > len(contexts):
            invalid.append(marker)
            continue
        scored = contexts[marker - 1]
        chunk = scored.chunk
        citations.append(
            Citation(
                marker=marker,
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                source=chunk.source,
                title=chunk.title,
                page=chunk.page,
                quote=_best_quote(chunk.text, focus.get(marker, answer)),
                score=scored.score,
            )
        )
    return citations, invalid


def renumber_answer(
    answer: str, contexts: Sequence[ScoredChunk]
) -> tuple[str, list[Citation], list[int]]:
    """Drop invalid markers from the prose and densify the survivors.

    Returns ``(rewritten_answer, citations, invalid_markers)`` where
    ``citations[i].marker == i + 1`` and every marker left in the text has a
    matching citation. Quotes are chosen against the *original* prose, before
    renumbering, so the surrounding sentence is still intact when it is used.
    """
    answer = normalize_markers(answer)
    citations, invalid = resolve_citations(answer, contexts)
    if not citations:
        # Nothing survives, so the only possible edit is deleting bad markers.
        return (_strip_all(answer) if invalid else answer), [], invalid

    mapping = {citation.marker: index + 1 for index, citation in enumerate(citations)}
    rewritten = _rewrite_markers(answer, mapping)
    renumbered = [
        citation.model_copy(update={"marker": mapping[citation.marker]})
        for citation in citations
    ]
    return rewritten, renumbered, invalid


def strip_markers(text: str) -> str:
    """Remove every marker and tidy the whitespace it leaves behind."""
    return _strip_all(text)


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #
def _strip_all(text: str) -> str:
    return _tidy(_MARKER_GROUP_RE.sub("", normalize_markers(text)))


def _rewrite_markers(text: str, mapping: dict[int, int]) -> str:
    """Renumber kept markers, delete the rest, collapsing empty groups."""
    dropped = False

    def _replace(match: re.Match[str]) -> str:
        nonlocal dropped
        kept: list[int] = []
        for part in match.group(1).split(","):
            renumbered = mapping.get(int(part.strip()))
            if renumbered is None:
                dropped = True
            elif renumbered not in kept:
                kept.append(renumbered)
        if not kept:
            return ""
        return "[" + ", ".join(str(number) for number in kept) + "]"

    rewritten = _MARKER_GROUP_RE.sub(_replace, text)
    # Only a deletion can leave a hole; a pure renumber must not reflow prose
    # the model wrote deliberately.
    return _tidy(rewritten) if dropped else rewritten


def _tidy(text: str) -> str:
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = _INTERIOR_RUN_RE.sub(" ", text)
    text = _TRAILING_SPACE_RE.sub("", text)
    return text.strip()


def _content_words(text: str) -> set[str]:
    return {token for token in tokenize(text) if token not in _STOPWORDS}


def _focus_sentences(answer: str) -> dict[int, str]:
    """Marker -> the marker-free answer sentence that cites it (first wins)."""
    focus: dict[int, str] = {}
    for sentence in split_sentences(answer):
        markers = extract_markers(sentence)
        if not markers:
            continue
        cleaned = _strip_all(sentence)
        for marker in markers:
            focus.setdefault(marker, cleaned)
    return focus


def _best_quote(chunk_text: str, focus: str, limit: int = QUOTE_MAX_CHARS) -> str:
    """Pick the chunk sentence that best covers ``focus``'s content words.

    Containment does the work - it asks how much of the claim the sentence
    accounts for - and a small jaccard term breaks ties towards the sentence
    that spends fewest words doing so, which keeps quotes tight. Iteration is
    in document order with a strict ``>``, so ties resolve to the earliest
    sentence and the result is stable across runs.
    """
    sentences = split_sentences(chunk_text) or [chunk_text]
    focus_words = _content_words(focus)
    best = sentences[0]
    if focus_words:
        best_score = -1.0
        for sentence in sentences:
            words = _content_words(sentence)
            if not words:
                continue
            score = containment(focus_words, words) + 0.25 * jaccard(focus_words, words)
            if score > best_score:
                best_score, best = score, sentence
    return truncate(normalize_whitespace(best), limit)
