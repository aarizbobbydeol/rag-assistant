"""FastAPI dependency wiring.

The pipeline is expensive to build (it loads an embedder and restores the index
from disk), so it is created once per process and shared. Everything else here
is a thin accessor so route handlers stay free of construction logic.
"""

from __future__ import annotations

import threading
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from app.auth import AuthService, User
from app.config import Settings, get_settings
from app.pipeline import RagPipeline

_pipeline: RagPipeline | None = None
_auth: AuthService | None = None
_lock = threading.Lock()


def get_pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        with _lock:
            if _pipeline is None:
                _pipeline = RagPipeline(get_settings())
    return _pipeline


def get_auth_service() -> AuthService:
    global _auth
    if _auth is None:
        settings = get_settings()
        with _lock:
            if _auth is None:
                _auth = AuthService(settings, settings.auth_db_path)
    return _auth


def reset_state() -> None:
    """Drop the singletons. Tests use this between cases; nothing else should."""
    global _pipeline, _auth
    with _lock:
        _pipeline = None
        _auth = None


SettingsDep = Annotated[Settings, Depends(get_settings)]
PipelineDep = Annotated[RagPipeline, Depends(get_pipeline)]
AuthDep = Annotated[AuthService, Depends(get_auth_service)]


def current_user(
    settings: SettingsDep,
    auth: AuthDep,
    authorization: Annotated[str | None, Header()] = None,
) -> User | None:
    """Resolve the bearer token.

    Returns ``None`` when auth is disabled so local development and the demo
    scripts need no token, but a *malformed* token is always an error — silently
    treating a bad token as anonymous would hide client bugs.
    """
    if not authorization:
        if settings.auth_required:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing bearer token.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return None

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header must be 'Bearer <token>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return auth.verify_token(token)


CurrentUser = Annotated[User | None, Depends(current_user)]


def require_user(user: CurrentUser) -> User:
    """For routes that always need an identity, regardless of auth_required."""
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


RequiredUser = Annotated[User, Depends(require_user)]
