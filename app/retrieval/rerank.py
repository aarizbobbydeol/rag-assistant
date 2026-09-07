"""Reordering the fused candidate list before it becomes the prompt.

Fusion only knows *ranks*: it can tell that both retrievers liked a chunk, but
not that the chunk actually answers the question. A reranker reads the query and
the candidate text together and is the cheapest large win in a RAG pipeline,
because everything downstream - the context budget, the citations, the
groundedness verdict - is decided by which five passages survive this step.

The default `HeuristicReranker` needs no model and no network, which is what
keeps the "runs with zero API keys" promise while still being a real reranker
rather than a pass-through. `CrossEncoderReranker` and `LLMReranker` are the
upgrades, and both degrade to the heuristic instead of failing a request: a
reranker is an optimisation, and an optimisation that can take down retrieval is
a liability.

Every implementation returns new `ScoredChunk` objects. The candidate list is
owned by `RagIndex`, which reuses it for MMR, so rewriting scores in place would
corrupt the caller's view of what the retrievers actually returned.
"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence

from app.config import Settings
from app.errors import ConfigurationError, LLMError
from app.generation.llm import LLMClient
from app.generation.prompts import build_rerank_messages
from app.models import ScoredChunk
from app.observability import get_logger
from app.retrieval.hybrid import min_max_normalise
from app.utils import containment, normalize_whitespace, tokenize

logger = get_logger(__name__)

__all__ = [
    "Reranker",
    "NoopReranker",
    "HeuristicReranker",
    "CrossEncoderReranker",
    "LLMReranker",
    "get_reranker",
]

# Function words carry no retrieval signal but would dominate a coverage ratio:
# "how long do refunds take" is four content words, not seven. Kept local to the
# module, as `app.generation.llm` and `app.generation.citations` each keep their
# own - the three lists answer different questions and drift apart on purpose.
_STOPWORDS = frozenset(
    """
    a about an and any are as at be been being but by can could did do does for from had has
    have how i if in into is it its just may me more most my no not of on or our over shall
    should so some such than that the their them then there these they this those to too
    under up was we were what when where which who whom why will with would you your
    """.split()
)


class Reranker(ABC):
    """Reorders retrieval candidates against the query and keeps the best `top_n`."""

    name: str = "reranker"

    @abstractmethod
    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        """Return at most `top_n` new `ScoredChunk`s, best first, ranked from 1.

        Implementations must not mutate `candidates` or the objects inside it.
        """

    def __repr__(self) -> str:  # pragma: no cover - debugging affordance
        return f"<{type(self).__name__} name={self.name}>"


def _emit(
    candidates: Sequence[ScoredChunk], scores: Sequence[float], top_n: int
) -> list[ScoredChunk]:
    """Sort by score, renumber, truncate, and copy - the shared tail of every reranker.

    Ties break on the candidate's incoming position, so a reranker that cannot
    separate two chunks defers to the retrievers rather than to whatever order
    `sorted` happened to produce. `deep=True` detaches the copied chunk's
    metadata dict as well, so a caller mutating a result cannot reach back into
    the index's own chunk objects.
    """
    limit = max(0, top_n)
    order = sorted(range(len(candidates)), key=lambda index: (-scores[index], index))
    return [
        candidates[index].model_copy(
            update={
                "score": float(scores[index]),
                "rerank_score": float(scores[index]),
                "rank": rank,
            },
            deep=True,
        )
        for rank, index in enumerate(order[:limit], start=1)
    ]


# --------------------------------------------------------------------------- #
# Pass-through
# --------------------------------------------------------------------------- #
class NoopReranker(Reranker):
    """Keeps the fusion order untouched; the control arm of the ablation sweep.

    It still copies, truncates and stamps `rerank_score`, so switching backends
    can never change the *shape* of what the pipeline receives - only the order.
    """

    name = "none"

    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        return [
            candidate.model_copy(
                update={"rerank_score": float(candidate.score), "rank": rank}, deep=True
            )
            for rank, candidate in enumerate(candidates[: max(0, top_n)], start=1)
        ]


# --------------------------------------------------------------------------- #
# Default: lexical evidence blended with the fusion prior
# --------------------------------------------------------------------------- #
# The blend below is the whole model, so it is written out rather than tuned:
#
#   coverage  0.45  How much of the question the passage accounts for. The
#                   dominant term, because a passage that mentions three of the
#                   four things you asked about is the one that answers you.
#   proximity 0.25  Whether those matches sit together. "refund" in paragraph
#                   one and "30 days" in paragraph nine is a topical passage;
#                   "refunds within 30 days" is an answer. This is the signal
#                   bag-of-words retrieval throws away, and the main reason this
#                   reranker beats the fusion order it is given.
#   fusion    0.25  The retrievers' own verdict, min-max normalised across the
#                   candidates so BM25 and cosine scales never leak in. Kept as
#                   a real term rather than a tiebreak: it is the only signal
#                   here that saw the whole corpus, so it carries the idf-like
#                   knowledge that a lexical overlap ratio cannot.
#   position  0.05  A mild nudge toward the head of a document, where titles,
#                   definitions and policy statements live. Deliberately small:
#                   it should settle near-ties, never overturn evidence.
#
# The weights sum to 1.0 and every term is in [0, 1], so the output score is a
# comparable [0, 1] number - which matters, because `settings.min_retrieval_score`
# is applied to whatever the reranker leaves in `score`.
_W_COVERAGE = 0.45
_W_PROXIMITY = 0.25
_W_FUSION = 0.25
_W_POSITION = 0.05

# ordinal 0 -> 1.00, 5 -> 0.71, 20 -> 0.38: a gentle slope, not a cliff.
_POSITION_DECAY = 0.08

# A verbatim phrase match is decisive, but only once there is a phrase to match:
# a single-word "query" appearing in the text is just coverage again.
_PHRASE_MIN_TERMS = 2


def _content_terms(text: str) -> list[str]:
    """Deduplicated, order-preserving content words of a query."""
    terms: list[str] = []
    for token in tokenize(text):
        if token in _STOPWORDS or token in terms:
            continue
        terms.append(token)
    return terms


def _min_window(positions: Sequence[tuple[int, str]], needed: int) -> int:
    """Width of the tightest span of tokens containing all `needed` matched terms."""
    counts: Counter[str] = Counter()
    best = positions[-1][0] - positions[0][0] + 1
    left = 0
    for position, term in positions:
        counts[term] += 1
        while len(counts) == needed:
            left_position, left_term = positions[left]
            best = min(best, position - left_position + 1)
            counts[left_term] -= 1
            if counts[left_term] == 0:
                del counts[left_term]
            left += 1
    return best


def _proximity(query_terms: Sequence[str], tokens: Sequence[str]) -> float:
    """How tightly the matched query terms cluster inside the passage, in [0, 1].

    Scored as `(matched / asked) * (matched / tightest window)`: the first factor
    stops two terms out of six from looking like a perfect hit, the second rewards
    a passage where those matches are adjacent. A single matched term has no
    spread to measure and scores zero - coverage already counts it.
    """
    wanted = set(query_terms)
    hits = [(index, token) for index, token in enumerate(tokens) if token in wanted]
    matched = {token for _index, token in hits}
    if len(matched) < 2:
        return 0.0
    window = _min_window(hits, len(matched))
    return (len(matched) / len(wanted)) * (len(matched) / max(window, len(matched)))


class HeuristicReranker(Reranker):
    """Query-term coverage + phrase proximity + position prior + fusion score.

    Lexical, deterministic and effectively free (a few hundred microseconds over
    a 24-candidate list), which is what makes it the default: it is the only
    backend that can run on every request of a service with no model and no
    credentials. See the weight table above this class for what each term buys.
    """

    name = "heuristic"

    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        if not candidates:
            return []

        terms = _content_terms(query)
        wanted = set(terms)
        phrase = normalize_whitespace(query).lower() if len(terms) >= _PHRASE_MIN_TERMS else ""
        fusion = min_max_normalise([candidate.score for candidate in candidates])

        scores: list[float] = []
        for candidate, prior in zip(candidates, fusion, strict=True):
            chunk = candidate.chunk
            tokens = tokenize(chunk.text)
            coverage = containment(wanted, set(tokens))
            if phrase and phrase in normalize_whitespace(chunk.text).lower():
                proximity = 1.0
            else:
                proximity = _proximity(terms, tokens) if wanted else 0.0
            position = 1.0 / (1.0 + _POSITION_DECAY * max(0, chunk.ordinal))
            scores.append(
                _W_COVERAGE * coverage
                + _W_PROXIMITY * proximity
                + _W_FUSION * prior
                + _W_POSITION * position
            )
        return _emit(candidates, scores, top_n)


# --------------------------------------------------------------------------- #
# Optional: a real cross-encoder
# --------------------------------------------------------------------------- #
def _sigmoid(value: float) -> float:
    """Logit -> [0, 1], written to survive the extremes a cross-encoder produces."""
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


class CrossEncoderReranker(Reranker):
    """`sentence-transformers` cross-encoder: the accurate, expensive backend.

    Scores every (query, passage) pair through one transformer forward pass, so
    it sees word order and negation that no bag-of-words heuristic can. The
    import and the model load happen here rather than at module scope, so the
    package stays optional and a missing wheel is a config error the factory can
    catch, not an ImportError at startup.
    """

    name = "cross-encoder"

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:
            raise ConfigurationError(
                "sentence-transformers is not installed",
                detail="pip install 'rag-assistant[local-models]', "
                "or set RAG_RERANK_BACKEND=heuristic",
            ) from exc
        try:
            self._model = CrossEncoder(model_name)
        except Exception as exc:  # bad model id, or no network on first load
            raise ConfigurationError(
                f"could not load cross-encoder model {model_name!r}", detail=str(exc)
            ) from exc
        self.model_name = model_name
        self.batch_size = max(1, batch_size)
        self._fallback = HeuristicReranker()

    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        pairs = [[query, candidate.chunk.text] for candidate in candidates]
        try:
            raw = self._model.predict(pairs, batch_size=self.batch_size)
            # Cross-encoder logits are unbounded; the rest of the pipeline reads
            # `score` as a [0, 1] relevance (min_retrieval_score, the API), so
            # squash rather than pass the raw margin through.
            scores = [_sigmoid(float(value)) for value in raw]
        except Exception as exc:  # a wedged native runtime must not fail a query
            logger.warning(
                "cross-encoder scoring failed, using the heuristic reranker",
                extra={"model": self.model_name, "reason": str(exc)},
            )
            return self._fallback.rerank(query, candidates, top_n)
        return _emit(candidates, scores, top_n)


# --------------------------------------------------------------------------- #
# Optional: ask the model itself
# --------------------------------------------------------------------------- #
# The prompt asks for scores on a 0-10 scale; everything downstream wants [0, 1].
_LLM_SCORE_SCALE = 10.0


def _parse_rerank_scores(text: str, count: int) -> dict[int, float] | None:
    """`[{"index": 2, "score": 9}, ...]` -> `{candidate index: score in [0, 1]}`.

    Defensive by construction: models wrap JSON in prose or a fenced block, so
    the outermost `[...]` is extracted before parsing, entries that are not
    usable objects are dropped, and out-of-range indices are ignored rather than
    trusted. Returns `None` when nothing usable survives, which is the caller's
    signal to fall back instead of reordering on one lucky entry.
    """
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end < start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except ValueError:
        return None
    if not isinstance(parsed, list):
        return None

    scores: dict[int, float] = {}
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        try:
            index = int(entry["index"])
            value = float(entry["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if not 1 <= index <= count or index - 1 in scores:
            continue
        if not math.isfinite(value):
            continue
        scores[index - 1] = min(1.0, max(0.0, value / _LLM_SCORE_SCALE))
    return scores or None


class LLMReranker(Reranker):
    """Listwise reranking by the generation model.

    The strongest backend when the model is strong, and the only one that
    understands the *intent* behind a question rather than its words. It is also
    the one most likely to misbehave - a refusal, a preamble, half a JSON array -
    so every failure path lands on the heuristic reranker. A degraded ranking is
    a worse answer; a raised exception is no answer at all.
    """

    name = "llm"

    def __init__(self, llm: LLMClient, fallback: Reranker | None = None) -> None:
        self.llm = llm
        self.fallback = fallback or HeuristicReranker()

    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        try:
            response = self.llm.complete(
                build_rerank_messages(query, candidates), temperature=0.0
            )
            scores = _parse_rerank_scores(response.text, len(candidates))
        except LLMError as exc:
            logger.warning(
                "llm reranker unavailable, using the heuristic reranker",
                extra={"reason": exc.message},
            )
            return self.fallback.rerank(query, candidates, top_n)
        except Exception as exc:  # a third-party client may raise anything
            logger.warning(
                "llm reranker raised, using the heuristic reranker",
                extra={"reason": f"{type(exc).__name__}: {exc}"},
            )
            return self.fallback.rerank(query, candidates, top_n)

        if scores is None:
            logger.warning("llm reranker returned no usable scores, using the heuristic")
            return self.fallback.rerank(query, candidates, top_n)

        # A candidate the model skipped scores 0.0 rather than inheriting a
        # fusion score on a different scale; `_emit`'s positional tiebreak then
        # leaves the skipped ones behind the judged ones in retrieval order.
        return _emit(
            candidates, [scores.get(index, 0.0) for index in range(len(candidates))], top_n
        )


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
_ALIASES = {
    "auto": "auto",
    "heuristic": "heuristic",
    "default": "heuristic",
    "lexical": "heuristic",
    "cross-encoder": "cross-encoder",
    "crossencoder": "cross-encoder",
    "ce": "cross-encoder",
    "llm": "llm",
    "none": "none",
    "noop": "none",
    "off": "none",
}


def get_reranker(settings: Settings, llm: LLMClient | None = None) -> Reranker:
    """Build the reranker named by `settings.rerank_backend`.

    `auto` resolves to the heuristic rather than probing for the best available
    backend, which is where this factory deliberately differs from
    `get_embedder`: the other two backends both spend something the operator has
    not agreed to spend - a model download at first start, or an LLM call on
    every query - and "auto" is the default value of the setting. Upgrading is an
    explicit choice; degrading never is, so a named backend that cannot be built
    falls back to the heuristic with a warning instead of refusing to start.

    An unrecognised name still raises: that is a typo in a deployment, and
    silently ignoring it would hide the mistake behind a working service.

    Whether reranking runs at all is `settings.rerank_enabled`, applied per
    request by `RagIndex.search`, so a caller can still override it per query.
    """
    requested = _ALIASES.get(settings.rerank_backend.strip().lower().replace("_", "-"))
    if requested is None:
        raise ConfigurationError(
            f"unknown rerank backend {settings.rerank_backend!r}",
            detail="Expected one of: auto, heuristic, cross-encoder, llm, none.",
        )

    if requested in ("auto", "heuristic"):
        return HeuristicReranker()
    if requested == "none":
        return NoopReranker()

    if requested == "cross-encoder":
        try:
            return CrossEncoderReranker(settings.rerank_model)
        except ConfigurationError as exc:
            logger.warning(
                "requested rerank backend unavailable, falling back to heuristic",
                extra={"backend": requested, "reason": exc.message},
            )
            return HeuristicReranker()

    # `llm`: an unusable client is caught here rather than once per request. The
    # extractive backend is usable for answering but cannot follow a scoring
    # instruction, so wrapping it would buy a guaranteed fallback per query -
    # `generative` is the capability flag that distinguishes the two.
    if llm is None or not llm.available or not llm.generative:
        logger.warning(
            "llm rerank backend requires a chat model, falling back to heuristic",
            extra={"provider": getattr(llm, "provider", "none")},
        )
        return HeuristicReranker()
    return LLMReranker(llm)
