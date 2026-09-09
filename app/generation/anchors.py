"""Corpus attestation of a question's discriminative anchors.

``query_term_coverage`` asks whether each of the question's content words appears
somewhere in the union of the retrieved chunks. Three failures hide inside that
sentence, and each detector here fixes exactly one:

* ``tokenize`` splits ``SEV-4`` into ``sev`` and ``4``, and ``content_words``
  then drops the digit for being too short. The only discriminative part of the
  identifier is deleted before coverage is computed, so coverage reads 1.0 on a
  question about a severity level the corpus explicitly says does not exist.
* Coverage is a union over the whole retrieved set, so two unrelated low-scoring
  chunks can each supply one word of ``annual maximum`` and satisfy it between
  them, while no single passage says anything about an annual maximum.
* Coverage is a bag of unigrams, so ``Chief Executive Officer`` is fully covered
  by ``Chief Financial Officer`` plus ``Chief Information Security Officer``.
  Only the combination is fabricated.

Every detector is deliberately low-recall and high-precision: each fires on well
under five percent of questions, so the union refuses far more rarely than a
threshold that has to be traded off against over-refusal.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from app.utils import STOPWORDS, stem, tokenize

__all__ = [
    "AnchorIndex",
    "AnchorMiss",
    "SCAFFOLD_TERMS",
    "build_anchor_index",
    "chunk_vocab",
    "missing_anchors",
]

# Interrogative scaffolding: the words a question uses to ask, not to name its
# subject. These are the dominant source of "missing" question terms on
# questions the corpus answers perfectly well ("how many", "how quickly"), so
# they are stripped before anchoring. A linguistic constant, not a config field.
SCAFFOLD_TERMS = frozenset(
    """
    how what when where which who whom whose why many much long soon often
    need needs needed see sees point does do did is are was were can could
    should would will list give tell explain describe happen happens
    """.split()
)

_NEGATION_CUES = frozenset(
    """no not never none neither nor without cannot cant doesn isn aren
    don wasn werent hasn havent""".split()
)

# The orphan-span detector exists to catch a missing *attribute value* - "the
# annual maximum on the dental plan", where the passage names the plan and
# states no maximum. Requiring one of these words in the span is what keeps it
# to that job. Without it the detector fires on ordinary verb phrases ("keep
# working", "take effect", "still acceptable") and refuses questions the corpus
# answers perfectly well.
#
# Note this cannot be a rarity threshold: measured over the sample corpus,
# "keep" and "still" have df=1 while "annual" and "maximum" have df=3, so
# filtering by document frequency removes the real catch and keeps every false
# positive. The distinction is grammatical, not statistical.
_ATTRIBUTE_TERMS = frozenset(
    stem(word)
    for word in (
        # quantities and ceilings
        "maximum minimum cap caps limit limits threshold thresholds target"
        " targets quota allowance budget deductible premium copay fee fees"
        " cost costs price rate charge"
        # durations and schedules
        " duration period deadline notice frequency interval window timeout"
        " retention expiry"
        # sizes and counts
        " size length count total number percentage percent share portion"
        # qualifiers that pair with the above
        " annual annually monthly weekly daily yearly hourly default"
        " standard base initial"
    ).split()
)

# How far to the left of an occurrence a negation cue still scopes over it.
_NEG_WINDOW = 6
_CLAUSE_BREAK = re.compile(r"[.;:!?]")

# Glued or hyphenated identifiers: SEV-4, v1, AES-256, FIDO2, TLS1.2. The
# trailing \b - rather than a lookahead forbidding a period - is what lets an
# identifier at the end of a sentence still be attested.
_IDENT = re.compile(r"\b([A-Za-z]{1,12})-?(\d{1,5}(?:\.\d{1,5})?)\b")

# Space-separated identifiers ("TLS 1.2") only read as identifiers when the
# prefix is a family the corpus already uses in glued form. Without that
# scoping, "within 5 minutes" would manufacture a ("within", "5") anchor.
_LOOSE = re.compile(r"\b([A-Za-z]{2,12}) (\d{1,5}(?:\.\d{1,5})?)\b")

# An all-caps prefix followed by a space and a version number is the other way
# corpora write identifier families ("TLS 1.2"), and it is how the family is
# learned when the glued form never appears.
_CAPS_FAMILY = re.compile(r"\b([A-Z]{2,12}) (\d{1,5}(?:\.\d{1,5})?)\b")

# Documents write a version one way and readers ask about it another: the API
# reference header says "Version 2.8" where the question says "v2.8". Without
# folding the two, the guard refuses a question the document answers on its
# very first line.
_PREFIX_ALIASES = {"version": "v", "ver": "v", "rev": "v", "revision": "v"}

_SUBTOKEN = re.compile(r"[_\-]")
_WORD = re.compile(r"[A-Za-z0-9_]+")


def _canonical_prefix(prefix: str) -> str:
    return _PREFIX_ALIASES.get(prefix.lower(), prefix.lower())


@dataclass(frozen=True)
class AnchorMiss:
    """One unattested anchor, carrying the detector that objected to it."""

    detector: str
    anchor: str

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.anchor} ({self.detector})"


@dataclass
class AnchorIndex:
    """What the corpus positively attests, built once per index generation."""

    identifiers: frozenset[tuple[str, str]] = frozenset()
    families: frozenset[str] = frozenset()
    bigrams: frozenset[tuple[str, str]] = frozenset()
    df: dict[str, int] = field(default_factory=dict)
    n_chunks: int = 0


def chunk_vocab(text: str) -> set[str]:
    """Stems of one chunk, with compound sub-tokens expanded.

    ``page_size`` survives tokenization as a single token, so a question about
    the "maximum page size" would look like it names something the chunk never
    says. Indexing the parts as well as the whole is a tokenizer fix, not a
    loosening of the guard.
    """
    vocab: set[str] = set()
    for token in tokenize(text):
        vocab.add(stem(token))
        for part in _SUBTOKEN.split(token):
            if len(part) > 2:
                vocab.add(stem(part))
    return vocab


def _clause_left(text: str, start: int) -> str:
    """Text to the left of ``start``, back to the nearest clause boundary."""
    breaks = [m.end() for m in _CLAUSE_BREAK.finditer(text, 0, start)]
    return text[breaks[-1] : start] if breaks else text[:start]


def _is_negated(text: str, start: int) -> bool:
    """Whether a negation cue scopes over the occurrence at ``start``.

    ``There is no SEV-4`` mentions SEV-4, so a plain presence test would treat
    it as attested and let the question through. Deprecation verbs are
    deliberately not cues: ``v1 was retired`` still attests v1.
    """
    left = tokenize(_clause_left(text, start))
    return any(token in _NEGATION_CUES for token in left[-_NEG_WINDOW:])


def _identifier_occurrences(text: str) -> Iterable[tuple[str, str, int]]:
    for match in _IDENT.finditer(text):
        yield _canonical_prefix(match.group(1)), match.group(2), match.start()


def _question_identifier_spans(question: str) -> dict[tuple[str, str], str]:
    """Map each identifier to the text the question actually used.

    Attestation is case-folded, but a refusal that says "sev-4" when the reader
    wrote "SEV-4" looks like it misread them. Quote them back to themselves.
    """
    spans: dict[tuple[str, str], str] = {}
    for match in _IDENT.finditer(question):
        spans.setdefault(
            (_canonical_prefix(match.group(1)), match.group(2)), match.group(0)
        )
    for match in _LOOSE.finditer(question):
        spans.setdefault(
            (_canonical_prefix(match.group(1)), match.group(2)), match.group(0)
        )
    return spans


def build_anchor_index(texts: Sequence[str]) -> AnchorIndex:
    """Index the anchors the corpus positively attests."""
    identifiers: set[tuple[str, str]] = set()
    families: set[str] = set()
    bigrams: set[tuple[str, str]] = set()
    df: Counter[str] = Counter()

    # Families are learned corpus-wide first: a family written glued in one
    # chunk has to license the spaced form in every other chunk, not only in
    # chunks that happen to come later.
    for text in texts:
        families.update(prefix for prefix, _, _ in _identifier_occurrences(text))
        families.update(_canonical_prefix(m.group(1)) for m in _CAPS_FAMILY.finditer(text))
        # "Version 2.8" is a version identifier whatever its casing, so a known
        # version word seeds its family directly rather than waiting for a
        # glued or all-caps spelling that a prose document may never use.
        families.update(
            _canonical_prefix(m.group(1))
            for m in _LOOSE.finditer(text)
            if m.group(1).lower() in _PREFIX_ALIASES
        )

    for text in texts:
        for prefix, digits, start in _identifier_occurrences(text):
            if not _is_negated(text, start):
                identifiers.add((prefix, digits))

        # Space-separated identifiers only count for a learned family.
        for match in _LOOSE.finditer(text):
            prefix = _canonical_prefix(match.group(1))
            if prefix in families and not _is_negated(text, match.start()):
                identifiers.add((prefix, match.group(2)))

        stems = [stem(t) for t in tokenize(text)]
        bigrams.update(zip(stems, stems[1:], strict=False))
        df.update(chunk_vocab(text))

    return AnchorIndex(
        identifiers=frozenset(identifiers),
        families=frozenset(families),
        bigrams=frozenset(bigrams),
        df=dict(df),
        n_chunks=len(texts),
    )


def _question_identifiers(question: str, index: AnchorIndex) -> list[tuple[str, str]]:
    """Identifiers the question names, read from the raw string.

    Tokenizing first would destroy the digit, which is the whole signal.
    """
    found = [(p, d) for p, d, _ in _identifier_occurrences(question)]
    found += [
        (_canonical_prefix(m.group(1)), m.group(2))
        for m in _LOOSE.finditer(question)
        if _canonical_prefix(m.group(1)) in index.families
    ]
    return found


def _anchor_tokens(question: str) -> list[str]:
    """Content tokens that could name the question's subject."""
    return [
        t
        for t in tokenize(question)
        if t not in STOPWORDS and t not in SCAFFOLD_TERMS and len(t) > 2
    ]


def _case_informative(question: str) -> bool:
    """Whether capitalisation in this question carries signal at all.

    Title Case and ALL CAPS questions capitalise words the writer never meant as
    a name, so the name-phrase detector stands down rather than inventing
    anchors out of the casing.
    """
    words = _WORD.findall(question)[1:]
    if not words:
        return False
    if all(w[:1].isupper() for w in words):
        return False
    return not any(w[:1].isupper() and w.lower() in STOPWORDS for w in words)


def _capitalised_runs(question: str) -> list[list[str]]:
    """Maximal runs of two or more non-initial capitalised words."""
    words = _WORD.findall(question)
    runs: list[list[str]] = []
    current: list[str] = []
    for word in words[1:]:
        if word[:1].isupper() and word.lower() not in STOPWORDS:
            current.append(word)
            continue
        if len(current) >= 2:
            runs.append(current)
        current = []
    if len(current) >= 2:
        runs.append(current)
    return runs


def _name_phrase_misses(question: str, index: AnchorIndex) -> list[AnchorMiss]:
    """Fire when no adjacent stemmed pair of a capitalised run is attested.

    Requiring the *whole* run to be attested is too strict - it refuses "Atlas
    API Free tier rate limit", which the corpus answers. Requiring only that
    some adjacent pair exists still refuses "Chief Executive Officer", because
    neither "chief executive" nor "executive officer" is ever written, while the
    real "Chief Financial Officer" attests both of its pairs.
    """
    if not _case_informative(question):
        return []
    misses = []
    for run in _capitalised_runs(question):
        stems = [stem(word.lower().strip("'-")) for word in run]
        pairs = list(zip(stems, stems[1:], strict=False))
        if pairs and not any(pair in index.bigrams for pair in pairs):
            misses.append(AnchorMiss("name_phrase", " ".join(run)))
    return misses


def _orphan_span_misses(
    question: str,
    chunk_vocabs: Sequence[set[str]],
    index: AnchorIndex,
    min_span: int,
) -> list[AnchorMiss]:
    """Fire on a contiguous run of question anchors absent from the pivot chunk.

    All three conditions are load-bearing. Contiguity separates "annual maximum"
    - adjacent, and the dental passage says neither - from "daily meal
    reimbursement cap", whose misses are scattered. Requiring the terms to exist
    elsewhere in the corpus keeps this a relevance signal rather than a second
    retrieval failure. Scoping to the pivot is what makes split evidence visible
    at all, since the union hides it.
    """
    if not chunk_vocabs:
        return []

    anchors = [stem(t) for t in _anchor_tokens(question)]
    if not anchors:
        return []

    def idf(token: str) -> float:
        return 1.0 / (1 + index.df.get(token, 0))

    pivot = max(
        chunk_vocabs,
        key=lambda vocab: sum(idf(a) for a in anchors if a in vocab),
    )

    def flush(run: list[str]) -> None:
        # An orphan run only counts as evidence when it is asking for an
        # attribute the passage does not carry. A run of ordinary verbs is a
        # phrasing accident, not a missing fact.
        if len(run) >= min_span and any(token in _ATTRIBUTE_TERMS for token in run):
            misses.append(AnchorMiss("orphan_span", " ".join(run)))

    misses: list[AnchorMiss] = []
    run: list[str] = []
    for token in anchors:
        if token not in pivot and index.df.get(token, 0) >= 1:
            run.append(token)
            continue
        flush(run)
        run = []
    flush(run)
    return misses


def missing_anchors(
    question: str,
    chunk_vocabs: Sequence[set[str]],
    index: AnchorIndex,
    *,
    detectors: Sequence[str] = ("identifier", "name_phrase", "orphan_span"),
    min_span: int = 2,
) -> list[AnchorMiss]:
    """Anchors the question names that the corpus does not positively attest."""
    if not question.strip() or index.n_chunks == 0:
        return []

    misses: list[AnchorMiss] = []
    enabled = set(detectors)

    if "identifier" in enabled:
        spans = _question_identifier_spans(question)
        for prefix, digits in _question_identifiers(question, index):
            if (prefix, digits) not in index.identifiers:
                spelled = spans.get((prefix, digits), f"{prefix}-{digits}")
                misses.append(AnchorMiss("identifier", spelled))

    if "name_phrase" in enabled:
        misses.extend(_name_phrase_misses(question, index))

    if "orphan_span" in enabled:
        misses.extend(_orphan_span_misses(question, chunk_vocabs, index, min_span))

    return misses
