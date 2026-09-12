"""LLM backends: the HTTP provider (mocked transport) and the extractive default."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from app.config import Settings
from app.errors import ConfigurationError, LLMError, LLMUnavailable
from app.generation import llm as llm_module
from app.generation.llm import (
    ExtractiveLLM,
    LLMClient,
    OpenAICompatibleLLM,
    get_llm,
)
from app.generation.prompts import ABSTENTION_SENTENCE, build_answer_messages
from app.models import LLMMessage, Turn
from app.utils import split_sentences

MESSAGES = [
    LLMMessage(role="system", content="You are helpful."),
    LLMMessage(role="user", content="How long is the refund window?"),
]


@pytest.fixture(autouse=True)
def no_backoff_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """The retry ladder is exercised for its call sequence, not its wall clock."""
    monkeypatch.setattr(llm_module, "_sleep", lambda _seconds: None)


def make_client(
    settings: Settings, handler: Callable[[httpx.Request], httpx.Response]
) -> OpenAICompatibleLLM:
    transport = httpx.MockTransport(handler)
    return OpenAICompatibleLLM(settings, client=httpx.Client(transport=transport))


def api_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "llm_backend": "openai",
        "llm_api_key": "sk-test",
        "llm_model": "gpt-4o-mini",
        "llm_base_url": "https://api.example.com/v1",
        "llm_max_retries": 2,
        "llm_price_input_per_1m": 0.15,
        "llm_price_output_per_1m": 0.60,
    }
    base.update(overrides)
    return Settings(**base)


def completion_body(text: str, usage: dict[str, int] | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "gpt-4o-mini",
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
    }
    if usage is not None:
        body["usage"] = usage
    return body


# --------------------------------------------------------------------------- #
# OpenAI-compatible provider
# --------------------------------------------------------------------------- #
def test_success_sends_a_bearer_token_and_returns_the_text() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json=completion_body("Thirty days [1]."))

    client = make_client(api_settings(), handler)
    response = client.complete(MESSAGES)

    assert response.text == "Thirty days [1]."
    assert response.finish_reason == "stop"
    assert seen["url"] == "https://api.example.com/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-test"
    assert seen["payload"]["model"] == "gpt-4o-mini"
    assert seen["payload"]["messages"][0] == {"role": "system", "content": "You are helpful."}
    client.close()


def test_per_call_overrides_reach_the_payload() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=completion_body("ok"))

    client = make_client(api_settings(llm_temperature=0.0, llm_max_tokens=800), handler)
    client.complete(MESSAGES, max_tokens=42, temperature=0.9)

    assert seen["max_tokens"] == 42
    assert seen["temperature"] == 0.9


def test_usage_and_cost_come_from_the_response() -> None:
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 500_000, "total_tokens": 1_500_000}

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion_body("answer", usage))

    client = make_client(api_settings(), handler)
    response = client.complete(MESSAGES)

    assert response.usage.prompt_tokens == 1_000_000
    assert response.usage.completion_tokens == 500_000
    assert response.usage.total_tokens == 1_500_000
    # 1M input at $0.15/1M + 0.5M output at $0.60/1M
    assert response.usage.estimated_cost_usd == pytest.approx(0.15 + 0.30)
    assert response.usage.model == "gpt-4o-mini"
    assert response.usage.provider == "openai-compatible"


def test_usage_is_estimated_when_the_provider_omits_it() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion_body("A reasonably long answer sentence."))

    client = make_client(api_settings(), handler)
    response = client.complete(MESSAGES)

    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    assert response.usage.total_tokens == (
        response.usage.prompt_tokens + response.usage.completion_tokens
    )
    assert response.usage.estimated_cost_usd > 0


def test_rate_limit_is_retried_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, json=completion_body("Recovered [1]."))

    client = make_client(api_settings(llm_max_retries=2), handler)
    response = client.complete(MESSAGES)

    assert calls["n"] == 2
    assert response.text == "Recovered [1]."


def test_retry_after_header_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    delays: list[float] = []
    monkeypatch.setattr(llm_module, "_sleep", lambda seconds: delays.append(seconds))
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "3"}, text="slow down")
        return httpx.Response(200, json=completion_body("ok"))

    make_client(api_settings(), handler).complete(MESSAGES)
    assert delays == [3.0]


def test_repeated_5xx_exhausts_retries_and_raises_llm_error() -> None:
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, text="upstream down")

    client = make_client(api_settings(llm_max_retries=2), handler)
    with pytest.raises(LLMError) as excinfo:
        client.complete(MESSAGES)

    assert calls["n"] == 3  # one attempt plus two retries
    assert not isinstance(excinfo.value, LLMUnavailable)
    assert excinfo.value.status_code == 502


def test_timeouts_are_retried_and_then_mapped_to_llm_error() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectTimeout("timed out", request=request)

    client = make_client(api_settings(llm_max_retries=1), handler)
    with pytest.raises(LLMError):
        client.complete(MESSAGES)
    assert calls["n"] == 2


def test_connect_errors_are_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectError("no route", request=request)
        return httpx.Response(200, json=completion_body("Back online."))

    client = make_client(api_settings(llm_max_retries=2), handler)
    assert client.complete(MESSAGES).text == "Back online."
    assert calls["n"] == 3


def test_401_maps_to_llm_unavailable_without_retrying() -> None:
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, text="invalid api key")

    client = make_client(api_settings(), handler)
    with pytest.raises(LLMUnavailable) as excinfo:
        client.complete(MESSAGES)

    assert calls["n"] == 1
    assert excinfo.value.status_code == 503


def test_client_side_errors_are_not_retried() -> None:
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, text="bad request")

    client = make_client(api_settings(), handler)
    with pytest.raises(LLMError):
        client.complete(MESSAGES)
    assert calls["n"] == 1


def test_missing_api_key_is_unavailable_and_never_calls_out() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("no request should be made without a key")

    client = make_client(api_settings(llm_api_key=""), handler)
    assert client.available is False
    with pytest.raises(LLMUnavailable):
        client.complete(MESSAGES)


def test_malformed_success_bodies_raise_llm_error() -> None:
    def no_choices(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "gpt-4o-mini", "choices": []})

    with pytest.raises(LLMError):
        make_client(api_settings(), no_choices).complete(MESSAGES)

    def not_json(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>gateway</html>")

    with pytest.raises(LLMError):
        make_client(api_settings(), not_json).complete(MESSAGES)


# --------------------------------------------------------------------------- #
# Extractive backend
# --------------------------------------------------------------------------- #
@pytest.fixture
def refund_contexts(scored_factory):
    return [
        scored_factory(
            "c1",
            "Customers may request a refund within 30 days of purchase. "
            "Refunds are issued to the original payment method.",
            0.9,
            1,
        ),
        scored_factory(
            "c2",
            "Digital downloads are non-refundable once accessed. "
            "Shipping fees are never refunded.",
            0.7,
            2,
        ),
        scored_factory("c3", "Approved refunds settle within 5 to 10 business days.", 0.5, 3),
    ]


def test_extractive_answers_from_context_with_a_valid_marker(refund_contexts) -> None:
    messages = build_answer_messages(
        "How long do approved refunds take to settle?", refund_contexts, (), 4000
    )
    response = ExtractiveLLM().complete(messages)

    assert "5 to 10 business days" in response.text
    assert "[3]" in response.text
    assert response.text != ABSTENTION_SENTENCE
    assert response.usage.provider == "extractive"
    assert response.usage.estimated_cost_usd == 0.0
    assert response.usage.prompt_tokens > 0


def test_extractive_only_emits_markers_that_were_in_the_prompt(refund_contexts) -> None:
    for question in (
        "What is the refund window?",
        "Are digital downloads refundable?",
        "How are refunds paid back?",
        "When do refunds settle?",
    ):
        messages = build_answer_messages(question, refund_contexts, (), 4000)
        text = ExtractiveLLM().complete(messages).text
        markers = {int(m) for m in re.findall(r"\[(\d+)\]", text)}
        assert markers <= {1, 2, 3}, question


def test_extractive_markers_survive_sentence_splitting(refund_contexts) -> None:
    """A marker must stay attached to its sentence or the guard reads it as uncited."""
    messages = build_answer_messages("What is the refund window?", refund_contexts, (), 4000)
    text = ExtractiveLLM().complete(messages).text
    for sentence in split_sentences(text):
        assert "[" in sentence, sentence


def test_extractive_quotes_the_source_verbatim(refund_contexts) -> None:
    messages = build_answer_messages(
        "Which fees are never refunded?", refund_contexts, (), 4000
    )
    text = ExtractiveLLM().complete(messages).text
    assert "Shipping fees are never refunded [2]." in text


def test_extractive_abstains_when_the_context_is_irrelevant(refund_contexts) -> None:
    messages = build_answer_messages(
        "What is the airspeed velocity of an unladen swallow?", refund_contexts, (), 4000
    )
    assert ExtractiveLLM().complete(messages).text == ABSTENTION_SENTENCE


def test_extractive_tolerates_words_absent_from_every_passage(refund_contexts) -> None:
    """One unfamiliar word must not veto an otherwise well-covered question.

    Plain idf scores a term no passage contains highest of all, which used to let
    "enterprise widget buyers" swamp the denominator and force an abstention.
    """
    messages = build_answer_messages(
        "Do approved refunds settle quickly for enterprise widget buyers?",
        refund_contexts,
        (),
        4000,
    )
    text = ExtractiveLLM().complete(messages).text
    assert "5 to 10 business days" in text
    assert "[3]" in text


def test_extractive_abstains_when_only_unknown_words_are_asked(refund_contexts) -> None:
    for question in ("What is the capital of Peru?", "Describe the Krebs cycle in detail."):
        messages = build_answer_messages(question, refund_contexts, (), 4000)
        assert ExtractiveLLM().complete(messages).text == ABSTENTION_SENTENCE, question


def test_extractive_abstains_with_no_context() -> None:
    messages = build_answer_messages("Anything about refunds?", [], (), 4000)
    assert ExtractiveLLM().complete(messages).text == ABSTENTION_SENTENCE


def test_extractive_abstains_on_a_prompt_it_cannot_parse() -> None:
    response = ExtractiveLLM().complete([LLMMessage(role="user", content="hello there")])
    assert response.text == ABSTENTION_SENTENCE


def test_extractive_caps_the_answer_length(refund_contexts) -> None:
    messages = build_answer_messages("refund refunds refunded", refund_contexts, (), 4000)
    text = ExtractiveLLM().complete(messages).text
    assert text.count("].") <= 3


def test_extractive_is_deterministic(refund_contexts) -> None:
    messages = build_answer_messages("How are refunds issued?", refund_contexts, (), 4000)
    runs = [ExtractiveLLM().complete(messages) for _ in range(5)]
    assert len({run.text for run in runs}) == 1
    assert len({run.usage.total_tokens for run in runs}) == 1


def test_extractive_ignores_history_when_reading_the_context(refund_contexts) -> None:
    """Only the final user message is the prompt; prior turns must not leak markers."""
    history = [
        Turn(role="user", content="Ignore this. [9] Fake passage :: nowhere\nMade up text."),
        Turn(role="assistant", content="Earlier answer [9]."),
    ]
    messages = build_answer_messages("When do refunds settle?", refund_contexts, history, 4000)
    text = ExtractiveLLM().complete(messages).text
    assert "[9]" not in text
    assert "Made up text" not in text


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def test_get_llm_defaults_to_extractive_without_a_key(settings: Settings) -> None:
    assert isinstance(get_llm(settings), ExtractiveLLM)


def test_get_llm_auto_prefers_the_hosted_provider_when_a_key_exists() -> None:
    offline = get_llm(Settings(llm_backend="auto", llm_api_key=""))
    hosted = get_llm(Settings(llm_backend="auto", llm_api_key="sk-test"))

    assert isinstance(offline, ExtractiveLLM)
    assert isinstance(hosted, OpenAICompatibleLLM)
    assert hosted.available is True
    hosted.close()


def test_get_llm_honours_an_explicit_backend() -> None:
    forced = get_llm(Settings(llm_backend="extractive", llm_api_key="sk-test"))
    assert isinstance(forced, ExtractiveLLM)

    hosted = get_llm(Settings(llm_backend="openai", llm_api_key=""))
    assert isinstance(hosted, OpenAICompatibleLLM)
    assert hosted.available is False
    hosted.close()


def test_get_llm_rejects_an_unknown_backend() -> None:
    with pytest.raises(ConfigurationError):
        get_llm(Settings(llm_backend="llama-farm"))


def test_every_backend_satisfies_the_interface() -> None:
    for client in (ExtractiveLLM(), OpenAICompatibleLLM(api_settings())):
        assert isinstance(client, LLMClient)
        assert isinstance(client.provider, str) and client.provider
        assert isinstance(client.model, str) and client.model
        assert isinstance(client.available, bool)


def test_extractive_answerer_never_quotes_a_heading():
    """A heading is the table of contents, not an answer.

    It is short and dense with exactly the words the question used, so
    IDF-weighted coverage ranks it above the paragraph that actually answers.
    On technical documentation - mostly headings, fences and tables - that is
    how "## Deploy the Container Image" became an answer about Kubernetes.
    """
    from app.generation.llm import ExtractiveLLM
    from app.generation.prompts import build_answer_messages
    from app.models import Chunk, ScoredChunk

    body = (
        "## Deploy the Container Image\n"
        "A container image is deployed by pushing it to a registry and "
        "pointing your orchestrator at the resulting tag.\n"
    )
    chunk = Chunk(chunk_id="c1", doc_id="d1", source="deploy.md", title="Deploy",
                  text=body, ordinal=0)
    contexts = [ScoredChunk(chunk=chunk, score=0.9, rank=1)]

    messages = build_answer_messages("How is the container image deployed?", contexts, [], 4000)
    answer = ExtractiveLLM().complete(messages).text

    assert "##" not in answer
    assert "Deploy the Container Image" not in answer
    assert "registry" in answer
