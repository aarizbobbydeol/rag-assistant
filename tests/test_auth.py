"""Tests for password storage, login and the hand-rolled HS256 tokens."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator

import pytest

from app.auth import (
    AuthService,
    DuplicateEmail,
    InvalidCredentials,
    InvalidEmail,
    InvalidSignature,
    InvalidToken,
    MalformedToken,
    RegistrationDisabled,
    TokenExpired,
    UnsupportedAlgorithm,
    WeakPassword,
    _b64d,
    _b64e,
    _json_bytes,
    hash_password,
    verify_password,
)
from app.config import Settings
from app.errors import RagError

EMAIL = "ada@example.com"
PASSWORD = "correct-horse-battery"


@pytest.fixture
def auth(settings: Settings) -> Iterator[AuthService]:
    service = AuthService(settings, settings.auth_db_path)
    try:
        yield service
    finally:
        service.close()


def service_with(settings: Settings, **overrides: object) -> AuthService:
    """A second service over the same database but a tweaked configuration."""
    return AuthService(settings.model_copy(update=overrides), settings.auth_db_path)


def resign(token: str, *, header: dict | None = None, payload: dict | None = None) -> str:
    """Rebuild a token from decoded parts, keeping the original signature."""
    parts = token.split(".")
    head = header if header is not None else json.loads(_b64d(parts[0]))
    body = payload if payload is not None else json.loads(_b64d(parts[1]))
    return f"{_b64e(_json_bytes(head))}.{_b64e(_json_bytes(body))}.{parts[2]}"


# --------------------------------------------------------------------------- #
# Password primitives
# --------------------------------------------------------------------------- #
def test_hashes_are_salted_and_verifiable() -> None:
    first = hash_password(PASSWORD)
    second = hash_password(PASSWORD)

    assert first != second, "a fresh salt must be drawn per hash"
    assert first.startswith("pbkdf2_sha256$260000$")
    assert verify_password(PASSWORD, first)
    assert verify_password(PASSWORD, second)
    assert not verify_password("wrong", first)


def test_verify_password_rejects_garbage_without_raising() -> None:
    assert not verify_password(PASSWORD, "")
    assert not verify_password(PASSWORD, "plaintext")
    assert not verify_password(PASSWORD, "md5$1$aa$bb")
    assert not verify_password(PASSWORD, "pbkdf2_sha256$notanint$aa$bb")


# --------------------------------------------------------------------------- #
# Registration and login
# --------------------------------------------------------------------------- #
def test_full_round_trip(auth: AuthService) -> None:
    user = auth.register(EMAIL, PASSWORD)
    assert user.email == EMAIL
    assert user.is_active
    assert user.id

    logged_in = auth.authenticate(EMAIL, PASSWORD)
    assert logged_in.id == user.id

    token, expires_in = auth.issue_token(logged_in)
    assert expires_in == auth.settings.auth_token_ttl_s

    verified = auth.verify_token(token)
    assert verified.id == user.id
    assert verified.email == EMAIL


def test_email_is_normalised_and_case_insensitive(auth: AuthService) -> None:
    user = auth.register("  Ada@Example.COM ", PASSWORD)

    assert user.email == EMAIL
    assert auth.authenticate("ADA@example.com", PASSWORD).id == user.id


def test_wrong_password_is_rejected(auth: AuthService) -> None:
    auth.register(EMAIL, PASSWORD)

    with pytest.raises(InvalidCredentials):
        auth.authenticate(EMAIL, PASSWORD + "!")


def test_unknown_email_raises_the_same_error_as_a_wrong_password(auth: AuthService) -> None:
    auth.register(EMAIL, PASSWORD)

    with pytest.raises(InvalidCredentials) as unknown:
        auth.authenticate("nobody@example.com", PASSWORD)
    with pytest.raises(InvalidCredentials) as wrong:
        auth.authenticate(EMAIL, "nope")

    assert type(unknown.value) is type(wrong.value)
    assert unknown.value.message == wrong.value.message


def test_malformed_email_does_not_leak_a_different_error(auth: AuthService) -> None:
    with pytest.raises(InvalidCredentials):
        auth.authenticate("not-an-email", PASSWORD)


def test_unknown_email_costs_about_the_same_as_a_wrong_password(auth: AuthService) -> None:
    """No timing oracle: both paths run exactly one PBKDF2 derivation.

    The bound is deliberately loose - this asserts the decoy hash exists at all,
    not a precise duration on a shared CI box.
    """
    auth.register(EMAIL, PASSWORD)

    def elapsed(email: str) -> float:
        started = time.perf_counter()
        with pytest.raises(InvalidCredentials):
            auth.authenticate(email, "definitely-wrong")
        return time.perf_counter() - started

    elapsed("nobody@example.com")  # warm the cached decoy hash
    unknown = min(elapsed("nobody@example.com") for _ in range(3))
    wrong = min(elapsed(EMAIL) for _ in range(3))

    assert 0.4 <= unknown / wrong <= 2.5


def test_duplicate_registration_is_rejected_case_insensitively(auth: AuthService) -> None:
    auth.register(EMAIL, PASSWORD)

    with pytest.raises(DuplicateEmail) as exc:
        auth.register("ADA@Example.com", PASSWORD)
    assert exc.value.status_code == 409


def test_short_password_is_rejected(auth: AuthService) -> None:
    too_short = "a" * (auth.settings.auth_min_password_length - 1)

    with pytest.raises(WeakPassword) as exc:
        auth.register(EMAIL, too_short)
    assert exc.value.status_code == 400

    with pytest.raises(InvalidCredentials):
        auth.authenticate(EMAIL, too_short)  # nothing was written


@pytest.mark.parametrize("bad", ["", "   ", "ada", "ada@", "@example.com", "ada@example", "a b@c.d"])
def test_invalid_email_shapes_are_rejected(auth: AuthService, bad: str) -> None:
    with pytest.raises(InvalidEmail):
        auth.register(bad, PASSWORD)


def test_registration_can_be_disabled(settings: Settings) -> None:
    service = service_with(settings, auth_allow_registration=False)
    try:
        with pytest.raises(RegistrationDisabled) as exc:
            service.register(EMAIL, PASSWORD)
        assert exc.value.status_code == 403
    finally:
        service.close()


def test_auth_errors_are_rag_errors(auth: AuthService) -> None:
    with pytest.raises(RagError):
        auth.authenticate(EMAIL, PASSWORD)


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #
def test_stored_row_holds_no_plaintext_password(auth: AuthService, settings: Settings) -> None:
    auth.register(EMAIL, PASSWORD)

    conn = sqlite3.connect(str(settings.auth_db_path))
    try:
        rows = conn.execute("SELECT * FROM users").fetchall()
    finally:
        conn.close()

    assert len(rows) == 1
    dumped = " ".join(str(value) for value in rows[0])
    assert PASSWORD not in dumped
    assert "pbkdf2_sha256$260000$" in dumped


def test_schema_survives_a_reopen(settings: Settings) -> None:
    first = AuthService(settings, settings.auth_db_path)
    try:
        user = first.register(EMAIL, PASSWORD)
    finally:
        first.close()

    second = AuthService(settings, settings.auth_db_path)
    try:
        assert second.authenticate(EMAIL, PASSWORD).id == user.id
    finally:
        second.close()


def test_db_path_defaults_to_settings(settings: Settings) -> None:
    service = AuthService(settings)
    try:
        assert service.db_path == settings.auth_db_path
    finally:
        service.close()


# --------------------------------------------------------------------------- #
# Tokens
# --------------------------------------------------------------------------- #
def test_expired_token_is_rejected(auth: AuthService, settings: Settings) -> None:
    user = auth.register(EMAIL, PASSWORD)
    short_lived = service_with(settings, auth_token_ttl_s=-1)
    try:
        token, expires_in = short_lived.issue_token(user)
    finally:
        short_lived.close()

    assert expires_in == -1
    with pytest.raises(TokenExpired):
        auth.verify_token(token)


def test_tampered_payload_is_rejected(auth: AuthService) -> None:
    user = auth.register(EMAIL, PASSWORD)
    token, _ = auth.issue_token(user)
    payload = json.loads(_b64d(token.split(".")[1]))
    payload["email"] = "attacker@example.com"

    with pytest.raises(InvalidSignature):
        auth.verify_token(resign(token, payload=payload))


def test_tampered_signature_is_rejected(auth: AuthService, settings: Settings) -> None:
    user = auth.register(EMAIL, PASSWORD)
    other = service_with(settings, auth_secret="a-different-signing-key")
    try:
        forged, _ = other.issue_token(user)
    finally:
        other.close()

    with pytest.raises(InvalidSignature):
        auth.verify_token(forged)


def test_alg_none_is_rejected(auth: AuthService) -> None:
    user = auth.register(EMAIL, PASSWORD)
    token, _ = auth.issue_token(user)

    with pytest.raises(UnsupportedAlgorithm):
        auth.verify_token(resign(token, header={"alg": "none", "typ": "JWT"}))
    with pytest.raises(UnsupportedAlgorithm):
        auth.verify_token(resign(token, header={"alg": "HS512", "typ": "JWT"}))

    # The classic unsigned form carries an empty signature segment.
    unsigned = resign(token, header={"alg": "none", "typ": "JWT"}).rsplit(".", 1)[0] + "."
    with pytest.raises(InvalidToken):
        auth.verify_token(unsigned)


@pytest.mark.parametrize(
    "bad",
    ["", "not-a-token", "a.b", "a.b.c.d", "..", "a.b.c", "!!!.!!!.!!!"],
)
def test_malformed_tokens_are_rejected(auth: AuthService, bad: str) -> None:
    with pytest.raises(MalformedToken):
        auth.verify_token(bad)


def test_token_without_expiry_is_rejected(auth: AuthService) -> None:
    user = auth.register(EMAIL, PASSWORD)
    token, _ = auth.issue_token(user)
    payload = json.loads(_b64d(token.split(".")[1]))
    payload.pop("exp")
    # Signed correctly on purpose: the rejection must come from the claim check.
    signing_input = resign(token, payload=payload).rsplit(".", 1)[0]
    forged = f"{signing_input}.{_b64e(auth._sign(signing_input))}"

    with pytest.raises(MalformedToken):
        auth.verify_token(forged)


def test_token_for_a_deactivated_user_is_rejected(auth: AuthService, settings: Settings) -> None:
    user = auth.register(EMAIL, PASSWORD)
    token, _ = auth.issue_token(user)

    conn = sqlite3.connect(str(settings.auth_db_path))
    try:
        conn.execute("UPDATE users SET is_active = 0 WHERE id = ?", (user.id,))
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(InvalidToken):
        auth.verify_token(token)
    with pytest.raises(InvalidCredentials):
        auth.authenticate(EMAIL, PASSWORD)


def test_token_for_a_deleted_user_is_rejected(auth: AuthService, settings: Settings) -> None:
    user = auth.register(EMAIL, PASSWORD)
    token, _ = auth.issue_token(user)

    conn = sqlite3.connect(str(settings.auth_db_path))
    try:
        conn.execute("DELETE FROM users WHERE id = ?", (user.id,))
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(InvalidToken):
        auth.verify_token(token)


def test_tokens_are_unique_per_issue(auth: AuthService) -> None:
    user = auth.register(EMAIL, PASSWORD)
    first, _ = auth.issue_token(user)
    second, _ = auth.issue_token(user)

    assert first != second, "the jti claim must make every token distinct"
    assert json.loads(_b64d(first.split(".")[1]))["sub"] == user.id
