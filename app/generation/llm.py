"""LLM providers behind one interface, including a no-API-key default.

Two implementations matter. `OpenAICompatibleLLM` talks to any endpoint that
speaks `/chat/completions` - OpenAI, Together, Groq, vLLM, Ollama's shim - and
is the production path. `ExtractiveLLM` answers by selecting the context
sentences that best cover the question and tagging each with its `[n]` marker;
it is why the whole system runs, tests and evaluates with no credentials and no
network, and why the citation and groundedness layers can be exercised
end-to-end in CI.
"""

from __future__ import annotations

import contextlib
import math
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any

import httpx

from app.config import Settings
from app.errors import ConfigurationError, LLMError, LLMUnavailable
from app.generation.prompts import (
    ABSTENTION_SENTENCE,
    CONTEXT_HEADER,
    QUESTION_PREFIX,
)
from app.models import LLMMessage, LLMResponse, Usage
from app.observability import LLM_COST, LLM_ERRORS, LLM_TOKENS, get_logger
from app.utils import (
    estimate_tokens,
    normalize_whitespace,
    split_sentences,
    stem,
    stem_tokens,
    tokenize,
    truncate,
)

logger = get_logger(__name__)

# Transient conditions worth another attempt: rate limiting and the 5xx family
# every proxy in front of an inference server emits under load.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_BACKOFF_BASE_S = 0.5
#: Ceiling on blind exponential growth, where the client is only guessing.
_BACKOFF_CAP_S = 8.0
#: Ceiling on an explicit ``Retry-After``, which is the server telling us when
#: its window reopens. Higher than the blind cap because it is information
#: rather than guesswork, but still bounded so a hostile header cannot hang a
#: request indefinitely.
_RETRY_AFTER_CAP_S = 65.0


def _sleep(seconds: float) -> None:
    """Indirection so tests can run the retry ladder without real delays."""
    time.sleep(seconds)


def _record(usage: Usage) -> None:
    LLM_TOKENS.labels(kind="prompt", model=usage.model).inc(usage.prompt_tokens)
    LLM_TOKENS.labels(kind="completion", model=usage.model).inc(usage.completion_tokens)
    if usage.estimated_cost_usd:
        LLM_COST.labels(model=usage.model).inc(usage.estimated_cost_usd)


# Providers advertise when a rate-limit window reopens in two different ways.
# `Retry-After` is the standard one; Groq and several other OpenAI-compatible
# endpoints never send it and use `x-ratelimit-reset-*` durations instead
# ("6.617s", "7h58m4.8s"). Reading only the standard header means every retry
# falls back to blind exponential backoff and lands before the window reopens.
# "ms" precedes "m" in the alternation deliberately: matched the other way,
# "500ms" reads as 500 minutes and the client sleeps for eight hours.
_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|h|m|s)")


def _parse_duration(raw: str) -> float | None:
    """Seconds from "30", "6.617s" or "7h58m4.8s"; None if unparseable."""
    text = raw.strip().lower()
    if not text:
        return None
    with contextlib.suppress(ValueError):
        return float(text)
    scale = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}
    parts = _DURATION_PART.findall(text)
    if not parts:
        return None
    return sum(float(value) * scale[unit] for value, unit in parts)


def _reset_after(headers: Any) -> float | None:
    """How long the server says to wait, by whichever convention it uses.

    The token window is checked before the request window because a
    tokens-per-minute budget is what a document-heavy prompt actually exhausts,
    and its reset is usually seconds away while the request reset can be hours.
    """
    for name in ("Retry-After", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        seconds = _parse_duration(headers.get(name, "") or "")
        if seconds is not None and seconds >= 0:
            return seconds
    return None


class LLMClient(ABC):
    """Chat completion seam. Implementations must be safe to share across calls."""

    provider: str = "none"
    model: str = "none"
    available: bool = False

    #: Whether this client can write *new* prose rather than only select it.
    #: Callers that need genuine rewriting - question condensation, LLM
    #: reranking - must check this rather than `available`: the extractive
    #: backend is always available and can still only quote the context back.
    generative: bool = False

    @abstractmethod
    def complete(
        self,
        messages: Sequence[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Return one completion, or raise `LLMError` / `LLMUnavailable`."""

    def __repr__(self) -> str:  # pragma: no cover - debugging affordance
        return f"<{type(self).__name__} provider={self.provider} model={self.model}>"


# --------------------------------------------------------------------------- #
# Hosted providers
# --------------------------------------------------------------------------- #
class OpenAICompatibleLLM(LLMClient):
    """Any `/chat/completions` endpoint, with bounded exponential backoff."""

    provider = "openai-compatible"
    generative = True

    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self.settings = settings
        self.model = settings.llm_model
        self._api_key = settings.llm_api_key.strip()
        self.available = bool(self._api_key)
        self._url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=settings.llm_timeout_s)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def complete(
        self,
        messages: Sequence[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        if not self.available:
            LLM_ERRORS.labels(kind="unavailable").inc()
            raise LLMUnavailable(
                "No LLM API key configured.",
                "Set RAG_LLM_API_KEY, or use RAG_LLM_BACKEND=extractive to run offline.",
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": (
                self.settings.llm_temperature if temperature is None else temperature
            ),
            "max_tokens": self.settings.llm_max_tokens if max_tokens is None else max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        data = self._post_with_retries(payload, headers)
        response = self._parse(data, messages)
        _record(response.usage)
        return response

    # -- transport ---------------------------------------------------------- #
    def _post_with_retries(
        self, payload: dict[str, Any], headers: dict[str, str]
    ) -> dict[str, Any]:
        attempts = max(0, self.settings.llm_max_retries) + 1
        last_detail = ""
        for attempt in range(attempts):
            try:
                response = self._client.post(self._url, json=payload, headers=headers)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                # ConnectError and friends all derive from TransportError; they
                # mean "the request never got an answer", which is retryable.
                last_detail = f"{type(exc).__name__}: {exc}"
                LLM_ERRORS.labels(kind="transport").inc()
                if attempt + 1 < attempts:
                    _sleep(self._backoff(attempt))
                    continue
                break

            status = response.status_code
            if status < 400:
                try:
                    return response.json()
                except ValueError as exc:
                    LLM_ERRORS.labels(kind="decode").inc()
                    raise LLMError(
                        "LLM returned a non-JSON response.", truncate(response.text, 400)
                    ) from exc

            body = truncate(response.text, 400)
            if status in (401, 403):
                LLM_ERRORS.labels(kind="auth").inc()
                raise LLMUnavailable(
                    f"LLM provider rejected the API key (HTTP {status}).", body
                )
            if status not in RETRYABLE_STATUS:
                LLM_ERRORS.labels(kind=f"http_{status}").inc()
                raise LLMError(f"LLM request failed with HTTP {status}.", body)

            last_detail = f"HTTP {status}: {body}"
            LLM_ERRORS.labels(kind=f"http_{status}").inc()
            if attempt + 1 < attempts:
                _sleep(self._backoff(attempt, response))
                continue

        logger.warning("llm_retries_exhausted", extra={"attempts": attempts})
        raise LLMError(
            f"LLM request failed after {attempts} attempt(s).", last_detail or None
        )

    def _backoff(self, attempt: int, response: httpx.Response | None = None) -> float:
        """Exponential backoff, but honour `Retry-After` when the server sets it.

        The two ceilings are deliberately different. ``_BACKOFF_CAP_S`` bounds
        *blind* exponential growth, where the client is guessing. ``Retry-After``
        is not a guess - it is the server saying when its window reopens - so
        clamping it to the guessing ceiling both ignores the instruction and
        guarantees the retry lands too early. A provider enforcing a
        tokens-per-minute budget routinely asks for longer than eight seconds,
        and waiting eight burns an attempt to be told the same thing again.

        Deliberately jitter-free: two identical runs must issue the same calls
        in the same order for the evaluation harness to be reproducible.
        """
        delay = min(_BACKOFF_CAP_S, _BACKOFF_BASE_S * (2**attempt))
        if response is None:
            return delay

        advertised = _reset_after(response.headers)
        if advertised is not None:
            delay = max(delay, min(_RETRY_AFTER_CAP_S, advertised))
        return delay

    # -- response ----------------------------------------------------------- #
    def _parse(self, data: dict[str, Any], messages: Sequence[LLMMessage]) -> LLMResponse:
        choices = data.get("choices") or []
        if not choices:
            LLM_ERRORS.labels(kind="empty").inc()
            raise LLMError("LLM response contained no choices.", truncate(str(data), 400))
        choice = choices[0] or {}
        text = ((choice.get("message") or {}).get("content") or "").strip()
        finish_reason = choice.get("finish_reason") or "stop"

        raw_usage = data.get("usage") or {}
        prompt_tokens = _as_int(raw_usage.get("prompt_tokens"))
        completion_tokens = _as_int(raw_usage.get("completion_tokens"))
        # Local servers routinely omit the usage block; estimating keeps cost
        # dashboards populated instead of silently reporting zero spend.
        if prompt_tokens <= 0:
            prompt_tokens = sum(estimate_tokens(m.content) for m in messages)
        if completion_tokens <= 0:
            completion_tokens = estimate_tokens(text)
        total_tokens = _as_int(raw_usage.get("total_tokens")) or (
            prompt_tokens + completion_tokens
        )

        usage = Usage(
            provider=self.provider,
            model=data.get("model") or self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=self._cost(prompt_tokens, completion_tokens),
        )
        return LLMResponse(text=text, usage=usage, finish_reason=finish_reason)

    def _cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        cost = (
            prompt_tokens * self.settings.llm_price_input_per_1m
            + completion_tokens * self.settings.llm_price_output_per_1m
        ) / 1_000_000
        return round(cost, 8)


def _as_int(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


# --------------------------------------------------------------------------- #
# Offline default
# --------------------------------------------------------------------------- #
_STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "could", "did", "do", "does", "for", "from", "had", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its", "may", "me", "my", "of", "on", "or", "our", "shall", "should", "so", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "those", "to", "was", "we", "were", "what", "when", "where", "which", "who", "whom", "why", "will", "with", "would", "you", "your"]
)

# Below this fraction of the question's (idf-weighted) content words a sentence
# is not an answer, it is a topically adjacent sentence - abstain instead.
# Structure, not prose. A markdown heading is the most dangerous of these: it is
# short and dense with exactly the words the question used, so IDF-weighted
# coverage ranks it above the paragraph that actually answers - and "## Deploy
# the Container Image" asserts nothing at all. Quoting one is the extractive
# equivalent of answering with the table of contents.
#
# check_groundedness already draws this line for answers ("Here is what I
# found:" is framing, not a claim); the answerer never did, and on technical
# documentation - which is mostly headings, fences and tables - that showed.
_NON_ASSERTIVE = re.compile(
    r"""^\s*(?:
        \#{1,6}\s                 # markdown heading
      | \{\s*\#[\w-]+\s*\}        # trailing {#anchor} left by a heading
      | ```                       # code fence marker
      | \|?\s*[:-]{3,}\s*\|       # table separator row
      | <[^>]+>\s*$               # a lone html tag
    )""",
    re.VERBOSE,
)

_MIN_OVERLAP = 0.28
# Extra sentences must be nearly as good as the best one to earn a place.
_RELATIVE_FLOOR = 0.6
_MAX_SENTENCES = 3
_MAX_SENTENCE_CHARS = 480
_NEAR_DUPLICATE = 0.75
# How much a query term that appears nowhere in the retrieved passages costs,
# as a multiple of log(1 + n). Above 1.0 an unanswerable question is refused
# even when its other words match something.
_UNSEEN_WEIGHT_MULTIPLIER = 0.5


class ExtractiveLLM(LLMClient):
    """Answers by quoting the context sentences that best cover the question.

    It is not a language model: it re-reads the numbered passages out of the
    prompt this package built, scores every sentence by how much of the
    question's content it accounts for, and returns the best few verbatim with
    their `[n]` markers. That makes it fully deterministic, free, and incapable
    of hallucinating - a marker it emits was, by construction, in the prompt.
    """

    provider = "extractive"
    generative = False

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings
        self.model = "extractive"
        self.available = True

    def complete(
        self,
        messages: Sequence[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        prompt = _last_user_content(messages)
        question, passages = _parse_prompt(prompt)
        text = self._answer(question, passages)

        prompt_tokens = sum(estimate_tokens(m.content) for m in messages)
        completion_tokens = estimate_tokens(text)
        usage = Usage(
            provider=self.provider,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            estimated_cost_usd=0.0,
        )
        _record(usage)
        return LLMResponse(text=text, usage=usage, finish_reason="stop")

    def _answer(self, question: str, passages: list[tuple[int, str]]) -> str:
        query_terms = _content_terms(question)
        if not query_terms or not passages:
            return ABSTENTION_SENTENCE

        weights = _idf_weights(query_terms, passages)
        total_weight = sum(weights[term] for term in query_terms)
        if total_weight <= 0:
            return ABSTENTION_SENTENCE

        scored: list[tuple[float, int, int, str, int]] = []
        for position, (marker, body) in enumerate(passages):
            for index, sentence in enumerate(split_sentences(body)):
                if _NON_ASSERTIVE.match(sentence):
                    continue
                terms = set(stem_tokens(sentence))
                if not terms:
                    continue
                covered = sum(weights[t] for t in query_terms if t in terms)
                scored.append((covered / total_weight, position, index, sentence, marker))

        if not scored:
            return ABSTENTION_SENTENCE

        # Sort by score, then by passage rank and sentence order, so ties always
        # resolve toward the higher-ranked passage and never toward set order.
        scored.sort(key=lambda row: (-row[0], row[1], row[2]))
        best = scored[0][0]
        if best < _MIN_OVERLAP:
            return ABSTENTION_SENTENCE

        floor = max(_MIN_OVERLAP, best * _RELATIVE_FLOOR)
        chosen: list[tuple[str, int]] = []
        seen: list[set[str]] = []
        for score, _position, _index, sentence, marker in scored:
            if score < floor or len(chosen) >= _MAX_SENTENCES:
                break
            terms = set(stem_tokens(sentence))
            if any(_overlap(terms, prior) >= _NEAR_DUPLICATE for prior in seen):
                continue
            chosen.append((sentence, marker))
            seen.append(terms)

        return " ".join(_cite(sentence, marker) for sentence, marker in chosen)


def _overlap(a: set[str], b: set[str]) -> float:
    """Symmetric near-duplicate test: containment in the smaller sentence."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _cite(sentence: str, marker: int) -> str:
    """Append `[n]` *inside* the final period.

    `split_sentences` treats a bracket after a period as the start of the next
    sentence, so a trailing "text. [2]" would orphan the marker and the
    groundedness guard would score the sentence as uncited.
    """
    body = truncate(normalize_whitespace(sentence), _MAX_SENTENCE_CHARS)
    return f"{body.rstrip().rstrip('.!?…').rstrip()} [{marker}]."


def _content_terms(text: str) -> list[str]:
    """Deduplicated, order-preserving content word *stems* (no stopwords).

    Stopwords are matched on the raw token - the stopword list is written in
    surface forms - and only the survivors are stemmed, so that a question
    asking about "paging" can still match a document that says "page".
    """
    terms: list[str] = []
    for token in tokenize(text):
        if token in _STOPWORDS or len(token) < 2:
            continue
        stemmed = stem(token)
        if stemmed not in terms:
            terms.append(stemmed)
    return terms


def _idf_weights(terms: Iterable[str], passages: Sequence[tuple[int, str]]) -> dict[str, float]:
    """Weight a query term by how few passages contain it.

    A term present in every passage cannot discriminate between them, so the
    sentence that merely repeats the topic word scores lower than the one
    carrying the rare word the question actually hinges on.

    A term in *no* passage keeps the highest weight, which is the whole
    abstention mechanism. "What is the policy on cryptocurrency payments?" against
    a corpus with no cryptocurrency in it must score low enough to refuse; if the
    unmatched term were discounted, the common words ("policy") would carry the
    match and the assistant would answer with a grounded but irrelevant sentence.
    Costing an absent term at full weight is what makes an unanswerable question
    fall below `_MIN_OVERLAP`, and `_UNSEEN_WEIGHT_MULTIPLIER` tunes how decisive
    a single unfamiliar word is - measured against the golden set's
    `false_answer_rate` and `over_refusal_rate`, which move in opposite
    directions as it changes.
    """
    vocabularies = [set(stem_tokens(body)) for _marker, body in passages]
    n = len(vocabularies)
    unseen = math.log(1 + n) * _UNSEEN_WEIGHT_MULTIPLIER
    weights: dict[str, float] = {}
    for term in terms:
        df = sum(1 for vocabulary in vocabularies if term in vocabulary)
        weights[term] = unseen if df == 0 else math.log(1 + n / (1 + df))
    return weights


def _last_user_content(messages: Sequence[LLMMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return ""


def _parse_prompt(content: str) -> tuple[str, list[tuple[int, str]]]:
    """Recover `(question, [(marker, text), ...])` from a built answer prompt.

    Mirrors `prompts.build_answer_messages`; anything that does not look like
    that rendering degrades to "no passages", which makes the caller abstain
    rather than answer from thin air.
    """
    if not content:
        return "", []

    question = ""
    context_part = content
    split_at = content.rfind(f"\n{QUESTION_PREFIX}")
    if split_at >= 0:
        question = content[split_at + len(QUESTION_PREFIX) + 1 :].strip()
        context_part = content[:split_at]
    header_at = context_part.find(CONTEXT_HEADER)
    if header_at >= 0:
        context_part = context_part[header_at + len(CONTEXT_HEADER) :]

    passages: list[tuple[int, str]] = []
    marker: int | None = None
    body: list[str] = []
    for line in context_part.splitlines():
        parsed = _parse_marker_line(line)
        if parsed is not None:
            if marker is not None:
                passages.append((marker, "\n".join(body).strip()))
            marker, body = parsed, []
        elif marker is not None:
            body.append(line)
    if marker is not None:
        passages.append((marker, "\n".join(body).strip()))

    return question, [(m, text) for m, text in passages if text]


def _parse_marker_line(line: str) -> int | None:
    """`[3] Title (p.2) :: source` -> 3, for any other line -> None."""
    stripped = line.strip()
    if not stripped.startswith("["):
        return None
    close = stripped.find("]")
    if close < 2:
        return None
    digits = stripped[1:close]
    return int(digits) if digits.isdigit() else None


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def get_llm(settings: Settings) -> LLMClient:
    """Pick a backend: `auto` uses the hosted provider only when a key exists."""
    backend = (settings.llm_backend or "auto").strip().lower()
    if backend == "auto":
        backend = "openai" if settings.llm_api_key.strip() else "extractive"
    if backend in ("extractive", "none", "offline"):
        return ExtractiveLLM(settings)
    if backend in ("openai", "openai-compatible", "http"):
        return OpenAICompatibleLLM(settings)
    raise ConfigurationError(
        f"Unknown llm_backend {settings.llm_backend!r}.",
        "Expected one of: auto, openai, extractive.",
    )
