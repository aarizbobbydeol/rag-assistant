"""Tests for `app.retrieval.embeddings`.

The interesting properties of the default embedder are the ones the rest of the
pipeline silently relies on: identical vectors across instances *and* across
processes (a persisted index is decoded by a different interpreter), exact
shape/dtype (the vector stores do one raw matmul), unit norm (cosine == dot
product) and a zero vector for blank text (an empty chunk must never be
retrievable).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.errors import ConfigurationError, LLMError
from app.retrieval import embeddings as emb
from app.retrieval.embeddings import (
    Embedder,
    HashingEmbedder,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
    get_embedder,
)

ROOT = Path(__file__).resolve().parent.parent

TEXTS = [
    "Customers may request a refund within 30 days of purchase.",
    "Digital downloads are non-refundable once accessed.",
    "Approved refunds settle within 5 to 10 business days.",
]

# (query, on-topic document, off-topic document). Enough pairs, drawn from
# unrelated domains, that the ordering assertion cannot pass by luck.
TOPICAL_TRIPLES = [
    (
        "How long do I have to request a refund?",
        "Customers may request a refund within 30 days of purchase.",
        "The mitochondrion converts nutrients into adenosine triphosphate.",
    ),
    (
        "What is the capital city of France?",
        "Paris is the capital and most populous city of France.",
        "Refunds are issued to the original payment method.",
    ),
    (
        "How do I reset my password?",
        "To reset your password, click the forgot password link on the sign-in page.",
        "Shipping fees are never refunded once the parcel has left the warehouse.",
    ),
    (
        "kubernetes pod scheduling",
        "The Kubernetes scheduler assigns pods to nodes based on resource requests.",
        "Vitamin D is synthesised in the skin under ultraviolet light.",
    ),
    (
        "espresso brewing temperature",
        "Brew espresso with water between 90 and 96 degrees Celsius.",
        "The quarterly earnings call is scheduled for the first Tuesday of May.",
    ),
    (
        "who wrote pride and prejudice",
        "Pride and Prejudice is a novel written by Jane Austen in 1813.",
        "Diesel engines compress air to a high temperature before injecting fuel.",
    ),
]


# --------------------------------------------------------------------------- #
# HashingEmbedder
# --------------------------------------------------------------------------- #
def test_hashing_is_deterministic_across_instances() -> None:
    first = HashingEmbedder(dim=256).embed_documents(TEXTS)
    second = HashingEmbedder(dim=256).embed_documents(TEXTS)
    assert np.array_equal(first, second)


def test_hashing_is_deterministic_across_processes() -> None:
    """Guards the `hashlib`-not-`hash()` rule: PYTHONHASHSEED is salted per run."""
    script = (
        "import json;"
        "from app.retrieval.embeddings import HashingEmbedder;"
        "print(json.dumps(HashingEmbedder(dim=64)"
        ".embed_query('refund policy for digital downloads').tolist()))"
    )
    out = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    other_process = np.asarray(json.loads(out.stdout), dtype=np.float32)
    here = HashingEmbedder(dim=64).embed_query("refund policy for digital downloads")
    assert np.array_equal(here, other_process)


def test_shapes_and_dtype() -> None:
    embedder = HashingEmbedder(dim=128)
    matrix = embedder.embed_documents(TEXTS)
    assert matrix.shape == (len(TEXTS), 128)
    assert matrix.dtype == np.float32

    vector = embedder.embed_query("refund window")
    assert vector.shape == (128,)
    assert vector.dtype == np.float32


def test_empty_batch_keeps_its_shape() -> None:
    assert HashingEmbedder(dim=32).embed_documents([]).shape == (0, 32)


def test_vectors_are_unit_norm() -> None:
    matrix = HashingEmbedder(dim=256).embed_documents(TEXTS)
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5)


def test_query_and_document_encodings_agree() -> None:
    embedder = HashingEmbedder(dim=256)
    assert np.array_equal(
        embedder.embed_query(TEXTS[0]), embedder.embed_documents([TEXTS[0]])[0]
    )


@pytest.mark.parametrize("blank", ["", " ", "\n\t  \r\n", " "])
def test_blank_input_is_a_zero_vector(blank: str) -> None:
    embedder = HashingEmbedder(dim=64)
    vector = embedder.embed_query(blank)
    assert vector.shape == (64,)
    assert not np.isnan(vector).any()
    assert np.count_nonzero(vector) == 0


def test_blank_rows_do_not_disturb_their_neighbours() -> None:
    matrix = HashingEmbedder(dim=128).embed_documents([TEXTS[0], "   ", TEXTS[1]])
    assert np.count_nonzero(matrix[1]) == 0
    assert np.allclose(np.linalg.norm(matrix[[0, 2]], axis=1), 1.0, atol=1e-5)


def test_identical_text_is_maximally_similar() -> None:
    matrix = HashingEmbedder(dim=512).embed_documents([TEXTS[0], TEXTS[0]])
    assert float(matrix[0] @ matrix[1]) == pytest.approx(1.0, abs=1e-5)


def test_sublinear_tf_keeps_a_repeated_document_in_place() -> None:
    """1 + log(tf) means duplicating a document barely moves its direction."""
    embedder = HashingEmbedder(dim=512)
    once, twice = embedder.embed_documents([TEXTS[0], TEXTS[0] + " " + TEXTS[0]])
    assert float(once @ twice) > 0.98


@pytest.mark.parametrize(
    ("query", "related", "unrelated"),
    TOPICAL_TRIPLES,
    ids=[triple[0][:24] for triple in TOPICAL_TRIPLES],
)
def test_query_prefers_the_topically_matching_document(
    query: str, related: str, unrelated: str
) -> None:
    embedder = HashingEmbedder(dim=1024)
    vector = embedder.embed_query(query)
    matched, other = embedder.embed_documents([related, unrelated])
    assert float(vector @ matched) > float(vector @ other)


def test_character_grams_survive_morphology() -> None:
    """Word and bigram features alone would score this pair at exactly zero."""
    embedder = HashingEmbedder(dim=1024)
    query = embedder.embed_query("refunding")
    doc, unrelated = embedder.embed_documents(["refunded", "sailboat"])
    assert float(query @ unrelated) == pytest.approx(0.0, abs=1e-6)
    assert float(query @ doc) > 0.1


def test_dim_must_be_sane() -> None:
    with pytest.raises(ConfigurationError):
        HashingEmbedder(dim=4)


# --------------------------------------------------------------------------- #
# Optional backends
# --------------------------------------------------------------------------- #
def _block_module(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """`None` in sys.modules makes `import name` raise ImportError."""
    monkeypatch.setitem(sys.modules, name, None)


def test_sentence_transformer_raises_when_package_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_module(monkeypatch, "sentence_transformers")
    with pytest.raises(ConfigurationError) as excinfo:
        SentenceTransformerEmbedder("some/model")
    assert "sentence-transformers" in str(excinfo.value)


def test_openai_embedder_requires_an_api_key() -> None:
    with pytest.raises(ConfigurationError):
        OpenAIEmbedder(model="text-embedding-3-small", api_key="")


def test_openai_embedder_known_model_dimension() -> None:
    embedder = OpenAIEmbedder(model="text-embedding-3-small", api_key="sk-test")
    assert embedder.dim == 1536
    assert embedder.name == "openai"


# --------------------------------------------------------------------------- #
# OpenAI transport, exercised through httpx's MockTransport
# --------------------------------------------------------------------------- #
def _mock_transport(monkeypatch: pytest.MonkeyPatch, handler) -> list[dict]:
    """Route every httpx.Client through `handler`; return the captured payloads."""
    import httpx

    seen: list[dict] = []
    real_client = httpx.Client

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return handler(request, seen)

    def factory(*_args, **kwargs):
        return real_client(
            transport=httpx.MockTransport(wrapped), timeout=kwargs.get("timeout", 5.0)
        )

    monkeypatch.setattr(httpx, "Client", factory)
    monkeypatch.setattr(emb.time, "sleep", lambda _s: None)
    return seen


def _embedding_response(request, _seen):
    import httpx

    payload = json.loads(request.content)
    data = [
        # Deliberately out of order: the client must sort by "index".
        {"index": i, "embedding": [float(len(text)), 1.0, 0.0, -2.0]}
        for i, text in reversed(list(enumerate(payload["input"])))
    ]
    return httpx.Response(200, json={"data": data, "model": payload["model"]})


def test_openai_embedder_batches_skips_blanks_and_normalises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _mock_transport(monkeypatch, _embedding_response)
    embedder = OpenAIEmbedder(
        model="custom-model", api_key="sk-test", batch_size=2, dim=4
    )

    matrix = embedder.embed_documents(["alpha", "  ", "beta", "gamma", "delta"])

    assert matrix.shape == (5, 4)
    assert [call["input"] for call in seen] == [
        ["alpha", "beta"],
        ["gamma", "delta"],
    ]
    assert np.count_nonzero(matrix[1]) == 0
    assert np.allclose(np.linalg.norm(matrix[[0, 2, 3, 4]], axis=1), 1.0, atol=1e-5)
    # Row 0 is "alpha" (5 chars) despite the reversed response ordering.
    assert matrix[0][0] == pytest.approx(5.0 / np.linalg.norm([5.0, 1.0, 0.0, -2.0]))


def test_openai_embedder_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    def flaky(request, seen):
        if len(seen) == 1:
            return httpx.Response(429, json={"error": "slow down"})
        return _embedding_response(request, seen)

    seen = _mock_transport(monkeypatch, flaky)
    embedder = OpenAIEmbedder(
        model="custom-model", api_key="sk-test", dim=4, max_retries=2
    )

    matrix = embedder.embed_documents(["alpha"])
    assert len(seen) == 2
    assert np.linalg.norm(matrix[0]) == pytest.approx(1.0, abs=1e-5)


def test_openai_embedder_gives_up_with_an_llm_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    _mock_transport(monkeypatch, lambda request, seen: httpx.Response(503, text="down"))
    embedder = OpenAIEmbedder(
        model="custom-model", api_key="sk-test", dim=4, max_retries=1
    )
    with pytest.raises(LLMError):
        embedder.embed_documents(["alpha"])


def test_openai_embedder_rejects_a_dimension_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_transport(monkeypatch, _embedding_response)
    embedder = OpenAIEmbedder(model="custom-model", api_key="sk-test", dim=8)
    with pytest.raises(ConfigurationError) as excinfo:
        embedder.embed_documents(["alpha"])
    assert "RAG_EMBEDDING_DIM" in str(excinfo.value.detail)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def test_get_embedder_default_is_hashing(settings: Settings) -> None:
    embedder = get_embedder(settings)
    assert isinstance(embedder, HashingEmbedder)
    assert isinstance(embedder, Embedder)
    assert embedder.name == "hashing"
    assert embedder.dim == settings.embedding_dim


def test_get_embedder_auto_falls_back_to_hashing(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _block_module(monkeypatch, "sentence_transformers")
    resolved = get_embedder(settings.model_copy(update={"embedding_backend": "auto"}))
    assert isinstance(resolved, HashingEmbedder)


def test_get_embedder_auto_tries_openai_only_with_a_key(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _block_module(monkeypatch, "sentence_transformers")
    resolved = get_embedder(
        settings.model_copy(
            update={"embedding_backend": "auto", "llm_api_key": "sk-test"}
        )
    )
    assert isinstance(resolved, OpenAIEmbedder)


def test_get_embedder_named_backend_falls_back_when_unavailable(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _block_module(monkeypatch, "sentence_transformers")
    resolved = get_embedder(
        settings.model_copy(update={"embedding_backend": "sentence-transformers"})
    )
    assert isinstance(resolved, HashingEmbedder)

    # No API key is just as much an "unavailable backend" as a missing package.
    resolved = get_embedder(settings.model_copy(update={"embedding_backend": "openai"}))
    assert isinstance(resolved, HashingEmbedder)


def test_get_embedder_rejects_an_unknown_backend(settings: Settings) -> None:
    with pytest.raises(ConfigurationError):
        get_embedder(settings.model_copy(update={"embedding_backend": "word2vec"}))
