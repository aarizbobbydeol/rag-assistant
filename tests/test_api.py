"""HTTP-level tests: routing, auth enforcement, upload limits, error shape."""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import deps
from app.config import Settings, get_settings
from app.main import create_app

HANDBOOK = (
    "# Refund policy\n\n"
    "Customers may request a refund within 30 days of purchase.\n"
    "Approved refunds settle within 5 to 10 business days.\n\n"
    "## Exceptions\n\n"
    "Digital downloads are non-refundable once accessed.\n"
)


def _client(settings: Settings) -> Iterator[TestClient]:
    deps.reset_state()
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as client:
        yield client
    deps.reset_state()


@pytest.fixture
def api_settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        index_dir=tmp_path / "index",
        corpus_dir=tmp_path / "corpus",
        upload_dir=tmp_path / "uploads",
        auth_db_path=tmp_path / "users.sqlite3",
        auth_secret="test-secret",
        auth_required=False,
        persist_index=False,
        llm_backend="extractive",
        embedding_backend="hashing",
        vector_store="numpy",
    )


@pytest.fixture
def client(api_settings: Settings, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr("app.deps.get_settings", lambda: api_settings)
    monkeypatch.setattr("app.main.get_settings", lambda: api_settings)
    yield from _client(api_settings)


@pytest.fixture
def loaded_client(client: TestClient) -> TestClient:
    response = client.post(
        "/api/ingest/upload",
        files=[("files", ("handbook.md", HANDBOOK.encode(), "text/markdown"))],
    )
    assert response.status_code == 200, response.text
    assert response.json()["total_chunks"] >= 1
    return client


# --------------------------------------------------------------------------- #
# Ops
# --------------------------------------------------------------------------- #
def test_live_does_not_require_the_index(client: TestClient) -> None:
    assert client.get("/live").json() == {"status": "alive"}


def test_health_reports_backends(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["embedder"]
    assert body["vector_store"]
    assert body["llm_provider"]


def test_metrics_endpoint_serves_prometheus_text(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "rag_" in response.text


def test_every_response_carries_a_trace_id(client: TestClient) -> None:
    response = client.get("/live")
    assert response.headers.get("X-Trace-Id")
    assert response.headers.get("X-Response-Time-ms")


def test_supplied_trace_id_is_echoed(client: TestClient) -> None:
    response = client.get("/live", headers={"X-Trace-Id": "abc123"})
    assert response.headers["X-Trace-Id"] == "abc123"


def test_openapi_schema_builds(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert "/api/chat" in schema["paths"]
    assert "/api/ingest/upload" in schema["paths"]


# --------------------------------------------------------------------------- #
# Ingest + chat
# --------------------------------------------------------------------------- #
def test_upload_then_chat_with_citations(loaded_client: TestClient) -> None:
    response = loaded_client.post(
        "/api/chat", json={"question": "How many days do customers have to request a refund?"}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["citations"], body["answer"]
    assert body["trace_id"]
    assert "groundedness" in body
    markers = [c["marker"] for c in body["citations"]]
    assert markers == list(range(1, len(markers) + 1))


def test_search_returns_ranked_chunks(loaded_client: TestClient) -> None:
    body = loaded_client.post("/api/search", json={"query": "refund", "top_k": 3}).json()
    assert body["results"]
    assert [r["rank"] for r in body["results"]] == list(range(1, len(body["results"]) + 1))


def test_documents_listing_and_delete(loaded_client: TestClient) -> None:
    listing = loaded_client.get("/api/ingest/documents").json()
    assert listing["documents"]
    doc_id = listing["documents"][0]["doc_id"]

    deleted = loaded_client.delete(f"/api/ingest/documents/{doc_id}").json()
    assert deleted["chunks_removed"] > 0
    assert loaded_client.get("/api/ingest/documents").json()["documents"] == []


def test_unsupported_upload_is_reported_not_fatal(client: TestClient) -> None:
    response = client.post(
        "/api/ingest/upload", files=[("files", ("evil.exe", b"MZ\x00\x00", "application/octet-stream"))]
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total_documents"] == 0
    assert body["skipped"]


def test_oversized_upload_is_rejected(client: TestClient, api_settings: Settings) -> None:
    api_settings.max_upload_mb = 1
    payload = io.BytesIO(b"x" * (2 * 1024 * 1024))
    response = client.post(
        "/api/ingest/upload", files=[("files", ("big.txt", payload, "text/plain"))]
    )
    assert response.status_code == 413
    assert response.json()["error"] == "payload_too_large"


def test_chat_on_empty_index_abstains(client: TestClient) -> None:
    body = client.post("/api/chat", json={"question": "anything"}).json()
    assert body["groundedness"]["abstained"] is True
    assert body["citations"] == []


def test_session_history_round_trip(loaded_client: TestClient) -> None:
    loaded_client.post("/api/chat", json={"question": "How long do refunds take?", "session_id": "s"})
    history = loaded_client.get("/api/sessions/s").json()
    assert len(history["turns"]) == 2
    assert history["turns"][0]["role"] == "user"

    assert loaded_client.delete("/api/sessions/s").json()["cleared"] is True
    assert loaded_client.get("/api/sessions/s").json()["turns"] == []


def test_chat_validates_its_input(client: TestClient) -> None:
    assert client.post("/api/chat", json={"question": ""}).status_code == 422
    assert client.post("/api/chat", json={}).status_code == 422
    assert client.post("/api/chat", json={"question": "hi", "top_k": 999}).status_code == 422


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
def test_register_login_and_me(client: TestClient) -> None:
    created = client.post(
        "/api/auth/register", json={"email": "dev@example.com", "password": "correct-horse"}
    )
    assert created.status_code == 201, created.text
    token = created.json()["access_token"]

    logged_in = client.post(
        "/api/auth/token", json={"email": "dev@example.com", "password": "correct-horse"}
    )
    assert logged_in.status_code == 200

    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["email"] == "dev@example.com"


def test_wrong_password_is_rejected(client: TestClient) -> None:
    client.post("/api/auth/register", json={"email": "a@b.com", "password": "correct-horse"})
    response = client.post("/api/auth/token", json={"email": "a@b.com", "password": "wrong-horse"})
    assert response.status_code == 401


def test_duplicate_registration_is_rejected(client: TestClient) -> None:
    body = {"email": "dup@example.com", "password": "correct-horse"}
    assert client.post("/api/auth/register", json=body).status_code == 201
    assert client.post("/api/auth/register", json=body).status_code == 409


def test_malformed_authorization_header_is_an_error(client: TestClient) -> None:
    response = client.get("/api/auth/status", headers={"Authorization": "Basic abc"})
    assert response.status_code == 401


def test_garbage_token_is_rejected(client: TestClient) -> None:
    response = client.get("/api/auth/me", headers={"Authorization": "Bearer not.a.jwt"})
    assert response.status_code == 401


def test_protected_routes_when_auth_is_required(
    api_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    api_settings.auth_required = True
    monkeypatch.setattr("app.deps.get_settings", lambda: api_settings)
    monkeypatch.setattr("app.main.get_settings", lambda: api_settings)

    for client in _client(api_settings):
        assert client.post("/api/chat", json={"question": "hi"}).status_code == 401
        assert client.get("/live").status_code == 200, "probes must stay open"

        registered = client.post(
            "/api/auth/register", json={"email": "u@example.com", "password": "correct-horse"}
        )
        assert registered.status_code == 201
        headers = {"Authorization": f"Bearer {registered.json()['access_token']}"}
        assert client.post("/api/chat", json={"question": "hi"}, headers=headers).status_code == 200


def test_config_endpoint_redacts_secrets(client: TestClient) -> None:
    body = client.get("/api/admin/config").json()
    assert body["auth_secret"] == "***set***"
    assert "test-secret" not in str(body)
