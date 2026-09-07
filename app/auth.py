"""Users, password hashing and HS256 tokens.

Everything here is stdlib on purpose: ``sqlite3`` for storage, ``hashlib`` +
``hmac`` for PBKDF2 and for the JWT signature. That keeps the auth surface small
enough to read end to end, and it means the service still boots with zero third
party crypto installed.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from app.config import Settings
from app.errors import RagError
from app.observability import get_logger

logger = get_logger(__name__)

# OWASP's 2023 floor for PBKDF2-HMAC-SHA256 is 600k rounds; 260k is the value
# fixed by the architecture doc, so it is what the stored tag records.
PBKDF2_ROUNDS = 260_000
PBKDF2_ALGORITHM = "pbkdf2_sha256"
SALT_BYTES = 16

JWT_ALGORITHM = "HS256"

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s.]+(?:\.[^@\s.]+)+$")
_MAX_EMAIL_LENGTH = 254

_dummy_hash_lock = threading.Lock()
_dummy_hash_cache: list[str] = []


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class AuthError(RagError):
    """Base class for everything this module raises."""

    status_code = 401
    code = "auth_error"


class InvalidCredentials(AuthError):
    """Wrong password *or* unknown account - deliberately indistinguishable."""

    status_code = 401
    code = "invalid_credentials"


class InvalidToken(AuthError):
    status_code = 401
    code = "invalid_token"


class MalformedToken(InvalidToken):
    code = "malformed_token"


class InvalidSignature(InvalidToken):
    code = "invalid_signature"


class UnsupportedAlgorithm(InvalidToken):
    code = "unsupported_algorithm"


class TokenExpired(InvalidToken):
    code = "token_expired"


class DuplicateEmail(AuthError):
    status_code = 409
    code = "duplicate_email"


class RegistrationDisabled(AuthError):
    status_code = 403
    code = "registration_disabled"


class InvalidEmail(AuthError):
    status_code = 400
    code = "invalid_email"


class WeakPassword(AuthError):
    status_code = 400
    code = "weak_password"


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class User(BaseModel):
    """A registered account, without any trace of its credentials.

    Lives here rather than in ``app.models`` so that nothing outside the auth
    layer is tempted to build one from unvalidated input.
    """

    id: str
    email: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    is_active: bool = True


# --------------------------------------------------------------------------- #
# Password hashing
# --------------------------------------------------------------------------- #
def _b64e(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64d(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def hash_password(password: str, *, rounds: int = PBKDF2_ROUNDS, salt: bytes | None = None) -> str:
    """Return an algorithm-tagged PBKDF2 hash: ``alg$rounds$salt$derived``.

    Tagging keeps the stored value self-describing, so the round count can be
    raised later without invalidating existing accounts.
    """
    if salt is None:
        salt = os.urandom(SALT_BYTES)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds)
    return f"{PBKDF2_ALGORITHM}${rounds}${_b64e(salt)}${_b64e(derived)}"


def verify_password(password: str, encoded: str) -> bool:
    """Constant-time check of ``password`` against a tagged hash."""
    try:
        algorithm, rounds, salt, expected = encoded.split("$")
        if algorithm != PBKDF2_ALGORITHM:
            return False
        candidate = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), _b64d(salt), int(rounds)
        )
    except (ValueError, binascii.Error):
        return False
    return hmac.compare_digest(_b64e(candidate), expected)


def _decoy_hash() -> str:
    """A real hash of a random secret, used when no account matches.

    Verifying against it makes the unknown-email path do the same PBKDF2 work as
    the wrong-password path, so response time does not leak which accounts
    exist. Computed once per process and cached: paying 260k rounds at import
    time would slow every ``import app.auth``.
    """
    with _dummy_hash_lock:
        if not _dummy_hash_cache:
            _dummy_hash_cache.append(hash_password(secrets.token_urlsafe(32)))
        return _dummy_hash_cache[0]


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #
_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    email         TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    created_at    REAL NOT NULL,
    is_active     INTEGER NOT NULL DEFAULT 1
);
CREATE UNIQUE INDEX IF NOT EXISTS users_email_lower_idx ON users (lower(email));
"""


class AuthService:
    """Registration, login and bearer tokens backed by one SQLite file."""

    def __init__(self, settings: Settings, db_path: Path | str | None = None) -> None:
        """``db_path`` defaults to ``settings.auth_db_path``.

        The architecture doc gives the path as an explicit argument and also
        calls it a setting; keeping the argument positional and optional honours
        both readings.
        """
        self.settings = settings
        self.db_path = Path(db_path if db_path is not None else settings.auth_db_path)
        self._lock = threading.Lock()
        self._conn = self._connect()
        self._ensure_schema()

    # -- storage -------------------------------------------------------- #
    def _connect(self) -> sqlite3.Connection:
        if self.db_path.parent and str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # FastAPI serves requests from a thread pool, so the connection is shared
        # across threads and every use is serialised by ``self._lock`` instead.
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._lock:
            try:
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=NORMAL")
            except sqlite3.Error:  # pragma: no cover - filesystem without WAL
                logger.warning("sqlite WAL unavailable, falling back to the default journal")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _row_by_email(self, email: str) -> sqlite3.Row | None:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT id, email, password_hash, created_at, is_active "
                "FROM users WHERE lower(email) = ?",
                (email,),
            )
            return cursor.fetchone()

    def _row_by_id(self, user_id: str) -> sqlite3.Row | None:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT id, email, password_hash, created_at, is_active FROM users WHERE id = ?",
                (user_id,),
            )
            return cursor.fetchone()

    @staticmethod
    def _to_user(row: sqlite3.Row) -> User:
        return User(
            id=row["id"],
            email=row["email"],
            created_at=datetime.fromtimestamp(row["created_at"], tz=UTC),
            is_active=bool(row["is_active"]),
        )

    # -- accounts ------------------------------------------------------- #
    def register(self, email: str, password: str) -> User:
        """Create an account. Emails are stored lowercased and are unique."""
        if not self.settings.auth_allow_registration:
            raise RegistrationDisabled("Registration is disabled.")
        normalised = self._normalise_email(email)
        minimum = self.settings.auth_min_password_length
        if len(password) < minimum:
            raise WeakPassword(f"Password must be at least {minimum} characters.")

        user = User(id=uuid.uuid4().hex, email=normalised, is_active=True)
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO users (id, email, password_hash, created_at, is_active) "
                    "VALUES (?, ?, ?, ?, 1)",
                    (
                        user.id,
                        user.email,
                        hash_password(password),
                        user.created_at.timestamp(),
                    ),
                )
                self._conn.commit()
        except sqlite3.IntegrityError as exc:
            raise DuplicateEmail("That email address is already registered.") from exc
        logger.info("user registered", extra={"user_id": user.id})
        return user

    def authenticate(self, email: str, password: str) -> User:
        """Verify credentials, raising one generic error for every failure.

        An unknown email is checked against a decoy hash so that it costs the
        same PBKDF2 work as a wrong password: no user-enumeration oracle by
        error message or by timing.
        """
        try:
            normalised = self._normalise_email(email)
        except InvalidEmail:
            normalised = email.strip().lower()
        row = self._row_by_email(normalised)
        encoded = row["password_hash"] if row is not None else _decoy_hash()
        matched = verify_password(password, encoded)
        if row is None or not matched or not row["is_active"]:
            raise InvalidCredentials("Invalid email or password.")
        user = self._to_user(row)
        logger.info("user authenticated", extra={"user_id": user.id})
        return user

    def get_user(self, user_id: str) -> User | None:
        row = self._row_by_id(user_id)
        return self._to_user(row) if row is not None else None

    # -- tokens --------------------------------------------------------- #
    def issue_token(self, user: User) -> tuple[str, int]:
        """Sign an HS256 JWT for ``user``; returns ``(token, expires_in_s)``."""
        ttl = int(self.settings.auth_token_ttl_s)
        issued_at = int(time.time())
        header = {"alg": JWT_ALGORITHM, "typ": "JWT"}
        payload = {
            "sub": user.id,
            "email": user.email,
            "iat": issued_at,
            "exp": issued_at + ttl,
            "jti": uuid.uuid4().hex,
        }
        signing_input = f"{_b64e(_json_bytes(header))}.{_b64e(_json_bytes(payload))}"
        signature = self._sign(signing_input)
        return f"{signing_input}.{_b64e(signature)}", ttl

    def verify_token(self, token: str) -> User:
        """Validate a bearer token and return the account it names.

        The user is re-read from storage rather than trusted from the claims, so
        a deactivated or deleted account cannot keep using an unexpired token.
        """
        parts = token.split(".") if token else []
        if len(parts) != 3 or not all(parts):
            raise MalformedToken("Token is not a well formed JWT.")
        encoded_header, encoded_payload, encoded_signature = parts

        try:
            header = json.loads(_b64d(encoded_header))
            signature = _b64d(encoded_signature)
        except (ValueError, binascii.Error) as exc:
            raise MalformedToken("Token header or signature is not decodable.") from exc
        if not isinstance(header, dict):
            raise MalformedToken("Token header is not an object.")
        # Checked before the signature so that "alg": "none" is rejected as such
        # rather than as an empty-signature mismatch.
        if header.get("alg") != JWT_ALGORITHM:
            raise UnsupportedAlgorithm(f"Unsupported token algorithm: {header.get('alg')!r}.")
        if header.get("typ") not in (None, "JWT"):
            raise MalformedToken("Unexpected token type.")

        expected = self._sign(f"{encoded_header}.{encoded_payload}")
        if not hmac.compare_digest(signature, expected):
            raise InvalidSignature("Token signature does not match.")

        try:
            payload = json.loads(_b64d(encoded_payload))
        except (ValueError, binascii.Error) as exc:
            raise MalformedToken("Token payload is not decodable.") from exc
        if not isinstance(payload, dict):
            raise MalformedToken("Token payload is not an object.")

        expires_at = payload.get("exp")
        subject = payload.get("sub")
        if not isinstance(expires_at, int | float) or isinstance(expires_at, bool):
            raise MalformedToken("Token has no usable expiry.")
        if not isinstance(subject, str) or not subject:
            raise MalformedToken("Token has no subject.")
        if time.time() >= expires_at:
            raise TokenExpired("Token has expired.")

        user = self.get_user(subject)
        if user is None or not user.is_active:
            raise InvalidToken("Token subject is not an active user.")
        return user

    def _sign(self, signing_input: str) -> bytes:
        return hmac.new(
            self.settings.auth_secret.encode("utf-8"),
            signing_input.encode("ascii"),
            hashlib.sha256,
        ).digest()

    # -- validation ----------------------------------------------------- #
    @staticmethod
    def _normalise_email(email: str) -> str:
        candidate = (email or "").strip().lower()
        if not candidate or len(candidate) > _MAX_EMAIL_LENGTH or not _EMAIL_RE.match(candidate):
            raise InvalidEmail("That does not look like an email address.")
        return candidate


def _json_bytes(payload: dict[str, object]) -> bytes:
    """Compact, key-sorted JSON so a given claim set always signs identically."""
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
