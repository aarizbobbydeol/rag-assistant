"""Post-generation hallucination guard: is every claim actually in the sources?

Citations prove that the model *pointed* at a passage, not that the passage
says what the sentence says. This module closes that gap by scoring each answer
sentence against the text it cites and reporting a per-answer fraction, which
the pipeline can turn into an abstention when ``strict_grounding`` is on.

The scoring is deliberately lexical-first. An embedding says two spans are
*about* the same thing; it cannot tell "digital downloads are non-refundable"
from "digital downloads are refundable", and a fluent paraphrase of a passage
that asserts something the passage never asserted scores high on cosine. Word
overlap with the cited chunk is the weaker semantic signal but the stronger
*evidential* one, so it carries most of the weight and the embedding only
nudges the result.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

from app.config import Settings
from app.generation.anchors import AnchorIndex, AnchorMiss, missing_anchors
from app.generation.citations import extract_markers, strip_markers
from app.models import Citation, Groundedness, ScoredChunk, SentenceSupport
from app.observability import get_logger
from app.retrieval.embeddings import Embedder
from app.utils import (
    containment,
    content_words,
    normalize_whitespace,
    split_sentences,
    stem,
    stem_tokens,
    tokenize,
)

logger = get_logger(__name__)

__all__ = [
    "sentence_support",
    "check_groundedness",
    "should_abstain",
    "query_term_coverage",
    "anchor_support",
    "money_slot_unfilled",
]


# Function words plus the discourse scaffolding an answer wraps its claims in.
# Attribution verbs ("according", "found", "says") and pointers ("here",
# "following") assert nothing on their own, so a sentence built only from this
# list - "Here is what I found:" - is framing, not a claim, and there is
# nothing about it to ground. Kept local rather than in `app.utils` because
# BM25 and the eval metrics deliberately want the full term stream.
_STOPWORDS = frozenset(
    ["a", "about", "above", "after", "all", "also", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "but", "by", "can", "could", "did", "do", "does", "doing", "done", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "let", "me", "might", "more", "most", "must", "my", "no", "nor", "not", "note", "now", "of", "off", "on", "once", "only", "or", "other", "our", "out", "over", "own", "please", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "us", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would", "you", "your", "according", "based", "follow", "following", "follows", "found", "know", "say", "said", "says", "see", "seen", "tell"]
)

# A claim assembled from two sources is weaker evidence than one lifted from a
# single passage, and pooling every chunk's vocabulary lets an invented sentence
# collect its words from unrelated passages - exactly the failure this guard
# exists to catch. So support is the best single chunk, never the union.
_UNIGRAM_WEIGHT = 0.65
_LEXICAL_WEIGHT = 0.7


def sentence_support(
    sentence: str,
    chunk_texts: Sequence[str],
    embedder: Embedder | None = None,
) -> float:
    """Score in ``[0, 1]`` for how well ``chunk_texts`` back ``sentence``.

    Lexical evidence is the containment of the sentence's content words in the
    best-matching chunk, blended with the containment of its content-word
    bigrams so that reordering or splicing the source costs something. When an
    ``embedder`` is supplied the cosine similarity to the same chunk is mixed
    in at ``1 - 0.7`` weight.

    Lexical dominates on purpose: cosine similarity measures aboutness, and a
    paraphrase can be maximally *about* a passage while asserting a fact the
    passage never stated. For citation checking, shared vocabulary with the
    cited text is the stronger evidence, so the embedding can only nudge a
    score, never manufacture one.

    Returns ``0.0`` for an empty sentence, an empty chunk list, or a sentence
    with no content words - there is nothing to ground either way.
    """
    claim = strip_markers(sentence)
    words = _content_words(claim)
    if not words or not chunk_texts:
        return 0.0

    unique = set(words)
    bigrams = _bigrams(words)
    lexical = max(_lexical_overlap(unique, bigrams, text) for text in chunk_texts)

    if embedder is None:
        return _clamp(lexical)
    cosine = _cosine_support(claim, chunk_texts, embedder)
    if cosine is None:
        return _clamp(lexical)
    return _clamp(_LEXICAL_WEIGHT * lexical + (1.0 - _LEXICAL_WEIGHT) * cosine)


def check_groundedness(
    answer: str,
    contexts: Sequence[ScoredChunk],
    citations: Sequence[Citation],
    invalid_markers: Sequence[int],
    settings: Settings,
    embedder: Embedder | None = None,
) -> Groundedness:
    """Verdict on whether ``answer`` is backed by the retrieved contexts.

    A sentence carrying markers is checked only against the chunks it cites -
    that is the claim the answer is making about its own sources. A sentence
    with no usable marker is checked against every context (a lower bar, since
    any passage may back it) and counted in ``uncited_sentences``. Sentences
    whose markers were *all* invalid fall into that second path too: an
    invented source number is no citation at all.

    Sentences with no content words are scaffolding ("Here is what I found:")
    and are left out of the count entirely rather than scored as unsupported.
    ``score`` is supported / total, and ``0.0`` when nothing was scorable.
    """
    invalid = list(dict.fromkeys(int(marker) for marker in invalid_markers))

    if _is_abstention(answer, settings):
        # A refusal is correct behaviour, not a hallucination; scoring its
        # prose against the context would mark the safest possible answer as
        # the least grounded one.
        return Groundedness(
            score=1.0,
            invalid_citations=invalid,
            abstained=True,
            reason="the answer is the configured abstention message",
        )

    cited_text = _text_by_marker(contexts, citations)
    all_texts = [scored.chunk.text for scored in contexts]

    supports: list[SentenceSupport] = []
    unsupported: list[str] = []
    uncited = 0

    for sentence in split_sentences(answer):
        claim = strip_markers(sentence)
        if not _content_words(claim):
            continue
        markers = extract_markers(sentence)
        targets = [cited_text[m] for m in markers if m in cited_text]
        if not targets:
            targets = all_texts
            uncited += 1
        score = sentence_support(claim, targets, embedder)
        supported = score >= settings.groundedness_threshold
        supports.append(
            SentenceSupport(
                sentence=sentence,
                supported=supported,
                support_score=round(score, 4),
                cited_markers=markers,
            )
        )
        if not supported:
            unsupported.append(sentence)

    total = len(supports)
    supported_count = total - len(unsupported)
    score = supported_count / total if total else 0.0

    result = Groundedness(
        score=round(score, 4),
        supported_sentences=supported_count,
        total_sentences=total,
        sentences=supports,
        unsupported=unsupported,
        invalid_citations=invalid,
        uncited_sentences=uncited,
    )
    if settings.strict_grounding and score < settings.min_answer_groundedness:
        result.abstained = True
        result.reason = (
            f"groundedness {score:.2f} is below the required "
            f"{settings.min_answer_groundedness:.2f}"
        )
        logger.info(
            "answer failed the groundedness bar",
            extra={
                "groundedness": result.score,
                "unsupported": len(unsupported),
                "invalid_citations": len(invalid),
            },
        )
    return result


# A question carrying one of these is asking for an amount of money. Compared
# in stem space, and deliberately excluding "rate": "what is the Free tier rate
# limit" asks about throughput, not price.
_PRICE_CUES = frozenset(
    stem(word)
    for word in (
        "cost", "costs", "price", "prices", "pricing", "fee", "fees",
        "charge", "charges", "expensive", "tariff",
    )
)

# "percent" is money-adjacent on purpose: "40 percent of the replacement price"
# is a real answer to a question about cost.
_MONEY_RE = re.compile(
    r"[$\u20ac\u00a3]|\b(?:dollars?|usd|eur|euros?|gbp|pounds?|cents?|percent|percentage)\b",
    re.IGNORECASE,
)

# "It is free" answers "what does it cost" completely, and contains no money
# token at all, so the slot check has to recognise it or it refuses the
# clearest answer in the corpus.
_FREE_RE = re.compile(
    r"\b(?:free of charge|at no (?:\w+ )?(?:cost|charge|premium|expense)"
    r"|no (?:\w+ )?(?:cost|charge|premium|fee|fees) (?:to|for) (?:the )?employee"
    r"|no (?:employee|additional|extra) (?:cost|charge|premium|fee|fees)"
    r"|complimentary|fully (?:covered|paid)|covered in full)\b",
    re.IGNORECASE,
)


def query_term_coverage(question: str, contexts: Sequence[ScoredChunk]) -> float:
    """Fraction of the question's content words present in the retrieved context.

    This is the backend-independent half of the hallucination guard, and it runs
    *before* generation. Groundedness alone cannot catch the characteristic RAG
    failure where the corpus has nothing on the topic, retrieval returns its
    least-bad guess, and the model answers with a sentence that is perfectly
    grounded and completely irrelevant - "what is the policy on cryptocurrency
    payments?" answered from a paragraph about taxi fares. Such an answer scores
    1.0 on groundedness because it *is* quoted from a source.

    Words are compared as stems so ordinary inflection does not read as an
    absent term. Returns 1.0 for a question with no content words, leaving the
    decision to the retrieval-score gate.
    """
    terms = {stem(word) for word in content_words(question)}
    if not terms:
        return 1.0
    vocabulary: set[str] = set()
    for context in contexts:
        vocabulary.update(stem_tokens(context.chunk.text))
    return len(terms & vocabulary) / len(terms)


def anchor_support(
    question: str,
    chunk_vocabs: Sequence[set[str]],
    anchor_index: AnchorIndex,
    settings: Settings,
) -> list[AnchorMiss]:
    """Anchors the question names that the corpus never positively attests.

    The pre-generation companion to :func:`query_term_coverage`, and the answer
    to the case that coverage structurally cannot see: the question's vocabulary
    is all present, but scattered across chunks, split across near-miss phrases,
    or - for ``SEV-4`` - present only inside an explicit denial. Coverage reads
    1.0 on every one of those and answers anyway.

    Returns the offending anchors so a refusal can be logged, and later worded,
    against the specific term that was missing.
    """
    if not settings.anchor_guard_enabled:
        return []
    return missing_anchors(
        question,
        chunk_vocabs,
        anchor_index,
        detectors=settings.anchor_detectors,
        min_span=settings.anchor_min_orphan_span,
    )


def money_slot_unfilled(question: str, answer: str, settings: Settings) -> bool:
    """True when a question asks a price and the answer names no money.

    Retrieval happily returns the tier table for "how much does the Enterprise
    tier cost per month" - the tiers are right there, with numbers - and the
    extractive answer quotes request-per-minute limits as though they were a
    price. Every part of that answer is grounded; none of it is money.

    ``percent`` counts as a money token deliberately: an answer like "40 percent
    of the replacement price" is a real answer to a question about cost.
    """
    if not settings.money_slot_guard:
        return False
    asked = {stem(word) for word in tokenize(question)}
    if not (asked & _PRICE_CUES):
        return False
    body = strip_markers(answer)
    # "Both are included at no employee premium" answers a price question
    # completely while containing no money token at all. Refusing it would turn
    # the clearest possible answer - it is free - into a refusal.
    if _FREE_RE.search(body):
        return False
    return not _MONEY_RE.search(body)


def should_abstain(top_score: float, settings: Settings) -> bool:
    """True when the best retrieved chunk is too weak to answer from at all.

    This is the pre-generation gate; ``check_groundedness`` is the post-
    generation one.
    """
    return top_score < settings.min_retrieval_score


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #
def _content_words(text: str) -> list[str]:
    """Claim-bearing tokens in order; order is what makes bigrams possible."""
    return [token for token in tokenize(text) if token not in _STOPWORDS]


def _bigrams(words: Sequence[str]) -> set[str]:
    return {f"{first} {second}" for first, second in zip(words, words[1:], strict=False)}


def _lexical_overlap(words: set[str], bigrams: set[str], chunk_text: str) -> float:
    """Blend unigram and content-word-bigram containment for one chunk.

    The bigram term is what separates a quote from a bag of the same words
    rearranged into a different claim; it is the minority term because a
    faithful restatement legitimately breaks adjacency.
    """
    chunk_words = _content_words(chunk_text)
    unigram = containment(words, set(chunk_words))
    if not bigrams:
        return unigram
    bigram = containment(bigrams, _bigrams(chunk_words))
    return _UNIGRAM_WEIGHT * unigram + (1.0 - _UNIGRAM_WEIGHT) * bigram


def _cosine_support(
    sentence: str, chunk_texts: Sequence[str], embedder: Embedder
) -> float | None:
    """Best cosine against any chunk, or ``None`` if the embedder failed.

    A remote embedding backend can time out, and a guard that raises would turn
    a usable answer into a 500. Degrading to lexical-only evidence is both safe
    and loud enough in the log.
    """
    try:
        query = embedder.embed_query(sentence)
        matrix = embedder.embed_documents(list(chunk_texts))
    except Exception:
        logger.warning(
            "embedder failed during the groundedness check; "
            "falling back to lexical evidence",
            exc_info=True,
        )
        return None
    if matrix.size == 0:
        return None
    # Vectors are L2-normalised by contract, so the dot product is the cosine.
    # Negatives mean "unrelated" here, not "contradictory", so they floor at 0.
    return max(0.0, float(np.max(matrix @ query)))


def _text_by_marker(
    contexts: Sequence[ScoredChunk], citations: Sequence[Citation]
) -> dict[int, str]:
    """Marker -> full chunk text, resolved through ``chunk_id``.

    Markers are resolved by id rather than by position because
    ``renumber_answer`` may already have densified them, at which point a
    marker no longer indexes into ``contexts``. The stored quote is the
    fallback for a citation whose chunk is not in the context list.
    """
    by_id = {scored.chunk.chunk_id: scored.chunk.text for scored in contexts}
    mapping: dict[int, str] = {}
    for citation in citations:
        mapping.setdefault(citation.marker, by_id.get(citation.chunk_id, citation.quote))
    return mapping


def _is_abstention(answer: str, settings: Settings) -> bool:
    return _canonical(answer) == _canonical(settings.abstain_message)


def _canonical(text: str) -> str:
    return normalize_whitespace(text).casefold()


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))
