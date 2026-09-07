"""Registration, token issue and identity echo."""

from __future__ import annotations

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from app.auth import User
from app.deps import AuthDep, CurrentUser, RequiredUser, SettingsDep

router = APIRouter(prefix="/auth", tags=["auth"])


class CredentialsRequest(BaseModel):
    # Plain ``str`` rather than ``EmailStr`` so the service needs no
    # email-validator dependency; AuthService.register does the shape check.
    email: str = Field(..., min_length=3, max_length=320)
    password: str = Field(..., min_length=1, max_length=1024)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class UserResponse(BaseModel):
    id: str
    email: str
    created_at: str
    is_active: bool

    @classmethod
    def of(cls, user: User) -> UserResponse:
        return cls(
            id=user.id,
            email=user.email,
            created_at=str(user.created_at),
            is_active=user.is_active,
        )


TokenResponse.model_rebuild()


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an account and return a token",
)
def register(body: CredentialsRequest, auth: AuthDep) -> TokenResponse:
    user = auth.register(body.email, body.password)
    token, expires_in = auth.issue_token(user)
    return TokenResponse(
        access_token=token, expires_in=expires_in, user=UserResponse.of(user)
    )


@router.post("/token", response_model=TokenResponse, summary="Exchange credentials for a token")
def token(body: CredentialsRequest, auth: AuthDep) -> TokenResponse:
    user = auth.authenticate(body.email, body.password)
    access_token, expires_in = auth.issue_token(user)
    return TokenResponse(
        access_token=access_token, expires_in=expires_in, user=UserResponse.of(user)
    )


@router.get("/me", response_model=UserResponse, summary="Identity behind the current token")
def me(user: RequiredUser) -> UserResponse:
    return UserResponse.of(user)


@router.get("/status", summary="Whether this deployment requires authentication")
def auth_status(settings: SettingsDep, user: CurrentUser) -> dict[str, object]:
    return {
        "auth_required": settings.auth_required,
        "registration_open": settings.auth_allow_registration,
        "authenticated": user is not None,
        "email": user.email if user else None,
    }
