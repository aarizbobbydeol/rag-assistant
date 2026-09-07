"""Document -> chunk splitters.

Every splitter works in *character span* space: it decides where to cut
``doc.text`` and returns ``(start, end)`` index pairs, never re-joined strings.
That is the only way the invariant

    doc.text[chunk.start_char:chunk.end_char] == chunk.text

can hold by construction, and citation quoting downstream depends on it.

Size policy
-----------
Sizes are token counts from :func:`app.utils.estimate_tokens`. Splitters pack up
to ``chunk_size - chunk_overlap`` tokens of new material so that the overlap
prepended afterwards still fits inside ``chunk_size`` (the sliding-window
``fixed`` splitter instead cuts windows of ``chunk_size`` that share their
tails, which comes to the same thing). Runs of text with no usable separator
are cut at character boundaries rather than emitted oversize, so the only chunk
that can exceed ``chunk_size`` is one that absorbed an undersized neighbour:

    estimate_tokens(chunk.text) <= chunk_size + min_chunk_tokens

is the documented tolerance, and it is what ``tests/test_chunking.py`` asserts.
In the other direction a chunk under ``min_chunk_tokens`` is folded into a
neighbour, so the only undersized chunk left is a document that is itself
shorter than the minimum - or, in the pathological case where both neighbours
are already full, the fragment itself, because the ceiling outranks the floor.
"""

from __future__ import annotations

import bisect
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.errors import ConfigurationError
from app.models import Chunk, Document, SplitterName
from app.utils import estimate_tokens, jaccard, split_sentences, stable_id, tokenize

_SPAN = tuple[int, int]

_WORD_RUN = re.compile(r"\S+")

# (separator, tail_keep): the piece before a match ends at ``idx + tail_keep``
# and the next piece starts there, so a sentence keeps its full stop while a
# markdown heading keeps its own "## " marker. Leading whitespace is trimmed
# off every span at the end, which is why the head needs no second offset.
_RECURSIVE_SEPARATORS: tuple[tuple[str, int], ...] = (
    ("\n## ", 0),
    ("\n# ", 0),
    ("\n\n", 0),
    ("\n", 0),
    (". ", 1),
    (" ", 0),
)


# --------------------------------------------------------------------------- #
# Span helpers
# --------------------------------------------------------------------------- #
def _trim(text: str, start: int, end: int) -> _SPAN:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _grow(text: str, start: int, ends: Sequence[int], lo: int, budget: int) -> int:
    """Largest ``i >= lo`` with ``estimate_tokens(text[start:ends[i]]) <= budget``.

    Binary search rather than a linear walk: token counts are monotone in the
    end offset, and a linear walk would re-tokenise the growing prefix once per
    candidate, which is quadratic on a long document. Returns ``lo - 1`` when
    even the shortest candidate is over budget.
    """
    if lo >= len(ends) or estimate_tokens(text[start : ends[lo]]) > budget:
        return lo - 1
    best, low, high = lo, lo + 1, len(ends) - 1
    while low <= high:
        mid = (low + high) // 2
        if estimate_tokens(text[start : ends[mid]]) <= budget:
            best, low = mid, mid + 1
        else:
            high = mid - 1
    return best


def _pack(text: str, spans: Sequence[_SPAN], budget: int) -> list[_SPAN]:
    """Greedily merge adjacent spans into groups of at most ``budget`` tokens.

    A group is represented by ``(first.start, last.end)``, so whatever separator
    sat between the pieces stays in the chunk text verbatim.
    """
    if not spans:
        return []
    ends = [end for _, end in spans]
    packed: list[_SPAN] = []
    i = 0
    while i < len(spans):
        j = max(i, _grow(text, spans[i][0], ends, i, budget))
        packed.append((spans[i][0], spans[j][1]))
        i = j + 1
    return packed


def _hard_cut(text: str, start: int, end: int, budget: int) -> list[_SPAN]:
    """Cut a run that has no usable separator left at character boundaries.

    Reached only for pathological input (a whitespace-free run worth more than
    ``chunk_size`` tokens). Splitting mid-word is ugly but bounded, which is
    preferable to handing an embedder a chunk of unbounded length.
    """
    spans: list[_SPAN] = []
    pos = start
    while pos < end:
        tokens = max(1, estimate_tokens(text[pos:end]))
        cut = min(end, pos + max(1, (end - pos) * budget // tokens))
        while cut - pos > 1 and estimate_tokens(text[pos:cut]) > budget:
            cut -= max(1, (cut - pos) // 8)
        cut = min(end, max(cut, pos + 1))
        spans.append((pos, cut))
        pos = cut
    return spans


def _split_words(text: str, start: int, end: int, budget: int) -> list[_SPAN]:
    """Fallback for a single piece that is over budget on its own."""
    words: list[_SPAN] = []
    for match in _WORD_RUN.finditer(text, start, end):
        span = (match.start(), match.end())
        if estimate_tokens(text[span[0] : span[1]]) > budget:
            words.extend(_hard_cut(text, span[0], span[1], budget))
        else:
            words.append(span)
    return _pack(text, words, budget)


def _split_on(text: str, start: int, end: int, sep: str, tail_keep: int) -> list[_SPAN]:
    """Contiguous pieces of ``text[start:end]`` delimited by ``sep``."""
    pieces: list[_SPAN] = []
    pos = search = start
    while True:
        idx = text.find(sep, search, end)
        if idx < 0:
            break
        cut = min(idx + tail_keep, end)
        if cut > pos:
            pieces.append((pos, cut))
            pos = cut
        search = max(idx + len(sep), search + 1)
    if pos < end:
        pieces.append((pos, end))
    return pieces


def _sentence_spans(text: str) -> list[_SPAN]:
    """Locate every sentence from :func:`split_sentences` back in ``text``.

    ``split_sentences`` only strips and splits, so each sentence it returns is a
    verbatim substring; scanning forward from a moving cursor therefore recovers
    exact offsets even when a sentence repeats.
    """
    spans: list[_SPAN] = []
    cursor = 0
    for sentence in split_sentences(text):
        idx = text.find(sentence, cursor)
        if idx < 0:
            continue
        spans.append((idx, idx + len(sentence)))
        cursor = idx + len(sentence)
    return spans


def _merge_undersized(
    text: str, spans: Sequence[_SPAN], min_tokens: int, limit: int
) -> list[_SPAN]:
    """Fold spans below ``min_tokens`` into a neighbour that still has room.

    The predecessor is preferred; a leading fragment (a bare heading above its
    first paragraph) has none, so it folds forward instead. ``limit`` is what
    makes this safe: merging into a chunk that is already full would cascade -
    a splitter that emits one short piece per sentence would otherwise collapse
    an entire document into a single chunk. When neither neighbour has room the
    fragment stays as it is, because the size ceiling matters more than the
    floor. A document shorter than the minimum keeps its single chunk; dropping
    it would lose the document from the index.
    """
    pending = list(spans)
    out: list[_SPAN] = []
    i = 0
    while i < len(pending):
        start, end = pending[i]
        i += 1
        if estimate_tokens(text[start:end]) >= min_tokens:
            out.append((start, end))
        elif out and estimate_tokens(text[out[-1][0] : end]) <= limit:
            out[-1] = (out[-1][0], end)
        elif i < len(pending) and estimate_tokens(text[start : pending[i][1]]) <= limit:
            pending[i] = (start, pending[i][1])
        else:
            out.append((start, end))
    return out


def _overlap_start(text: str, start: int, end: int, overlap_tokens: int) -> int:
    """Word offset inside ``[start, end)`` whose suffix is <= ``overlap_tokens``.

    Returns ``end`` (no overlap) when even the final word is over budget, so the
    size guarantee wins over the overlap guarantee.
    """
    if overlap_tokens <= 0:
        return end
    starts = [match.start() for match in _WORD_RUN.finditer(text, start, end)]
    if not starts:
        return end
    best, low, high = end, 0, len(starts) - 1
    while low <= high:
        mid = (low + high) // 2
        if estimate_tokens(text[starts[mid] : end]) <= overlap_tokens:
            best, high = starts[mid], mid - 1
        else:
            low = mid + 1
    return best


def _apply_overlap(text: str, spans: Sequence[_SPAN], overlap_tokens: int) -> list[_SPAN]:
    """Extend each chunk backwards into the tail of the one before it.

    The back-off is measured against the *original* predecessor span, so the
    overlap never compounds across a long document, and it stops at the
    predecessor's second word so that starts stay strictly increasing even when
    a whole short chunk would fit inside the overlap budget.
    """
    if overlap_tokens <= 0 or len(spans) < 2:
        return list(spans)
    out = [spans[0]]
    for i in range(1, len(spans)):
        start, end = spans[i]
        prev_start, prev_end = spans[i - 1]
        floor = _second_word(text, prev_start, prev_end)
        back = _overlap_start(text, floor, min(prev_end, start), overlap_tokens)
        out.append((min(back, start), end))
    return out


def _second_word(text: str, start: int, end: int) -> int:
    """Offset of the second word in ``[start, end)``, or ``end`` if there is none."""
    words = _WORD_RUN.finditer(text, start, end)
    next(words, None)
    match = next(words, None)
    return match.start() if match is not None else end


class _PageMap:
    """Character offset -> page number, from ``Document.metadata["page_spans"]``."""

    __slots__ = ("_pages", "_starts")

    def __init__(self, doc: Document) -> None:
        parsed: list[tuple[int, int, int]] = []
        for entry in doc.metadata.get("page_spans") or []:
            try:
                start, end, page = entry[0], entry[1], entry[2]
                parsed.append((int(start), int(end), int(page)))
            except (TypeError, ValueError, IndexError, KeyError):
                continue  # a malformed loader hint must not break ingestion
        parsed.sort()
        self._pages = parsed
        self._starts = [start for start, _, _ in parsed]

    def page_for(self, pos: int) -> int | None:
        if not self._pages:
            return None
        idx = bisect.bisect_right(self._starts, pos) - 1
        # A chunk that begins in the gap between two page spans belongs to the
        # page it continues from, so clamp instead of returning None.
        return self._pages[max(idx, 0)][2]


# --------------------------------------------------------------------------- #
# Splitters
# --------------------------------------------------------------------------- #
class Splitter(ABC):
    """Base class: subclasses only decide *where* to cut.

    Merging undersized chunks, applying overlap and building the `Chunk` models
    are shared here so that every splitter satisfies the same invariants.
    """

    name: str = ""

    # Sliding windows carry their own overlap; everything else gets it applied.
    _native_overlap: bool = False

    def __init__(self, chunk_size: int, chunk_overlap: int, min_chunk_tokens: int = 24) -> None:
        if chunk_size < 1:
            raise ConfigurationError("chunk_size must be >= 1", f"got {chunk_size}")
        if chunk_overlap < 0:
            raise ConfigurationError("chunk_overlap must be >= 0", f"got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ConfigurationError(
                "chunk_overlap must be smaller than chunk_size",
                f"chunk_size={chunk_size} chunk_overlap={chunk_overlap}",
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_tokens = max(0, min_chunk_tokens)
        # Budget for *new* material: the overlap glued on afterwards has to fit
        # inside chunk_size too.
        self._budget = max(1, chunk_size - chunk_overlap)

    @abstractmethod
    def _segment(self, text: str) -> list[_SPAN]:
        """Base chunk spans, before undersize merging and overlap."""

    def split(self, doc: Document) -> list[Chunk]:
        text = doc.text
        if not text.strip():
            return []
        spans = [_trim(text, start, end) for start, end in self._segment(text)]
        spans = [(start, end) for start, end in spans if start < end]
        spans = _merge_undersized(
            text, spans, self.min_chunk_tokens, self._budget + self.min_chunk_tokens
        )
        if not self._native_overlap:
            spans = _apply_overlap(text, spans, self.chunk_overlap)
        return self._build(doc, spans)

    def _build(self, doc: Document, spans: Sequence[_SPAN]) -> list[Chunk]:
        pages = _PageMap(doc)
        chunks: list[Chunk] = []
        for ordinal, (start, end) in enumerate(spans):
            text = doc.text[start:end]
            chunks.append(
                Chunk(
                    chunk_id=stable_id(doc.doc_id, str(ordinal), text[:64]),
                    doc_id=doc.doc_id,
                    source=doc.source,
                    title=doc.title,
                    text=text,
                    ordinal=ordinal,
                    start_char=start,
                    end_char=end,
                    page=pages.page_for(start),
                    token_count=estimate_tokens(text),
                    metadata={"splitter": self.name},
                )
            )
        return chunks

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}, min_chunk_tokens={self.min_chunk_tokens})"
        )


class RecursiveSplitter(Splitter):
    """Structure-aware default: cut on the coarsest separator that fits.

    Tries headings, then paragraphs, then lines, then sentences, then words,
    descending only into the pieces that are still too large. Pieces that do fit
    are packed with their neighbours at the level where they were found, so a
    chunk boundary lands on the coarsest structural break available.
    """

    name = "recursive"
    separators: tuple[tuple[str, int], ...] = _RECURSIVE_SEPARATORS

    def _segment(self, text: str) -> list[_SPAN]:
        return self._descend(text, 0, len(text), 0)

    def _descend(self, text: str, start: int, end: int, level: int) -> list[_SPAN]:
        if start >= end:
            return []
        if estimate_tokens(text[start:end]) <= self._budget:
            return [(start, end)]
        if level >= len(self.separators):
            return _hard_cut(text, start, end, self._budget)

        sep, tail_keep = self.separators[level]
        pieces = _split_on(text, start, end, sep, tail_keep)
        if len(pieces) < 2:
            return self._descend(text, start, end, level + 1)

        out: list[_SPAN] = []
        fitting: list[_SPAN] = []
        for piece_start, piece_end in pieces:
            if estimate_tokens(text[piece_start:piece_end]) <= self._budget:
                fitting.append((piece_start, piece_end))
                continue
            out.extend(_pack(text, fitting, self._budget))
            fitting = []
            out.extend(self._descend(text, piece_start, piece_end, level + 1))
        out.extend(_pack(text, fitting, self._budget))
        return out


class FixedSplitter(Splitter):
    """Sliding window of ``chunk_size`` tokens stepping by the overlap.

    Windows start and end on word boundaries; the shared tail is what makes
    consecutive chunks overlap, so the generic overlap pass is skipped.
    """

    name = "fixed"
    _native_overlap = True

    def __init__(self, chunk_size: int, chunk_overlap: int, min_chunk_tokens: int = 24) -> None:
        super().__init__(chunk_size, chunk_overlap, min_chunk_tokens)
        # A window already contains its own overlap, so it gets the full budget
        # instead of reserving room for one to be prepended later.
        self._budget = chunk_size

    def _segment(self, text: str) -> list[_SPAN]:
        words = [(m.start(), m.end()) for m in _WORD_RUN.finditer(text)]
        if not words:
            return []
        starts = [start for start, _ in words]
        ends = [end for _, end in words]

        spans: list[_SPAN] = []
        i = 0
        while i < len(words):
            last = _grow(text, starts[i], ends, i, self._budget)
            if last < i:
                spans.extend(_hard_cut(text, starts[i], ends[i], self._budget))
                i += 1
                continue
            spans.append((starts[i], ends[last]))
            if last + 1 >= len(words):
                break
            back = _overlap_start(text, starts[i], ends[last], self.chunk_overlap)
            # bisect keeps the window contiguous; max() guarantees progress.
            i = max(i + 1, bisect.bisect_left(starts, back))
        return spans


class SentenceSplitter(Splitter):
    """Packs whole sentences, never cutting one in half unless it is oversize."""

    name = "sentence"

    def _segment(self, text: str) -> list[_SPAN]:
        out: list[_SPAN] = []
        fitting: list[_SPAN] = []
        for start, end in _sentence_spans(text):
            if estimate_tokens(text[start:end]) <= self._budget:
                fitting.append((start, end))
                continue
            out.extend(_pack(text, fitting, self._budget))
            fitting = []
            out.extend(_split_words(text, start, end, self._budget))
        out.extend(_pack(text, fitting, self._budget))
        return out


class SemanticSplitter(Splitter):
    """Packs sentences and breaks where the vocabulary turns over.

    Consecutive sentences that continue a topic reuse nouns and function words,
    which puts their token-set Jaccard around 0.2-0.4; a topic shift drops it
    towards zero. Breaking below ``similarity_threshold`` therefore keeps a
    coherent passage in one chunk without needing an embedding model - and it is
    deterministic, which an embedding-based breakpoint would not be across
    backends.

    Consequence worth knowing before selecting it: on terse, low-redundancy prose
    (release notes, specifications) almost every sentence pair scores near zero,
    so chunks land close to ``min_chunk_tokens`` rather than close to
    ``chunk_size``. Raise ``min_chunk_tokens`` when using this splitter.
    """

    name = "semantic"

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        min_chunk_tokens: int = 24,
        similarity_threshold: float = 0.15,
    ) -> None:
        super().__init__(chunk_size, chunk_overlap, min_chunk_tokens)
        self.similarity_threshold = similarity_threshold

    def _segment(self, text: str) -> list[_SPAN]:
        out: list[_SPAN] = []
        group: list[_SPAN] = []
        previous: set[str] = set()

        def flush() -> None:
            if group:
                out.append((group[0][0], group[-1][1]))
                group.clear()

        for start, end in _sentence_spans(text):
            current = set(tokenize(text[start:end]))
            if estimate_tokens(text[start:end]) > self._budget:
                flush()
                out.extend(_split_words(text, start, end, self._budget))
                previous = current
                continue
            if group:
                grown = estimate_tokens(text[group[0][0] : end])
                # An empty token set (punctuation, a bare number) says nothing
                # about topic, so only a real vocabulary turnover breaks - and
                # only once the chunk is worth keeping, since a break the
                # undersize merge would immediately undo is not a break.
                shifted = (
                    bool(previous)
                    and bool(current)
                    and jaccard(previous, current) < self.similarity_threshold
                    and estimate_tokens(text[group[0][0] : group[-1][1]]) >= self.min_chunk_tokens
                )
                if grown > self._budget or shifted:
                    flush()
            group.append((start, end))
            previous = current
        flush()
        return out


SPLITTERS: dict[str, type[Splitter]] = {
    RecursiveSplitter.name: RecursiveSplitter,
    FixedSplitter.name: FixedSplitter,
    SentenceSplitter.name: SentenceSplitter,
    SemanticSplitter.name: SemanticSplitter,
}


def get_splitter(
    name: str | SplitterName,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_tokens: int = 24,
) -> Splitter:
    """Build the splitter registered under ``name``.

    Accepts either the enum or the raw string: `Settings.splitter` is a
    `SplitterName`, whose hash is the member name rather than its value, so the
    value has to be unwrapped before the registry lookup.
    """
    key = str(getattr(name, "value", name)).strip().lower()
    try:
        splitter_cls = SPLITTERS[key]
    except KeyError:
        raise ConfigurationError(
            f"Unknown splitter: {key!r}",
            f"available: {', '.join(sorted(SPLITTERS))}",
        ) from None
    return splitter_cls(chunk_size, chunk_overlap, min_chunk_tokens)
