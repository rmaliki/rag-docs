"""Bearer-token validation: Microsoft Entra ID in production, signed dev tokens locally.

The user's identity, groups and roles come only from a token whose signature,
issuer, audience and expiry have been verified, never from request parameters.
"""

import time
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import quote

import httpx
import jwt
from jwt import PyJWKClient

from app.config import Settings

DEV_ISSUER = "rag-docs-dev"
DEV_AUDIENCE = "rag-docs"


class AuthError(Exception):
    """The bearer token is malformed, expired, or not issued for this API."""


@dataclass(frozen=True)
class AuthenticatedUser:
    object_id: str
    name: str | None = None
    username: str | None = None
    groups: tuple[str, ...] = ()
    roles: tuple[str, ...] = ()
    # Entra omits the `groups` claim when a user is in too many groups (the "overage"
    # case); membership must then be fetched from Microsoft Graph.
    groups_overage: bool = False


class TokenValidator(Protocol):
    def validate(self, token: str) -> AuthenticatedUser: ...


def _user_from_claims(claims: dict[str, Any]) -> AuthenticatedUser:
    claim_names = claims.get("_claim_names") or {}
    return AuthenticatedUser(
        object_id=claims["oid"],
        name=claims.get("name"),
        username=claims.get("preferred_username"),
        groups=tuple(claims.get("groups", ())),
        roles=tuple(claims.get("roles", ())),
        groups_overage="groups" in claim_names,
    )


class EntraTokenValidator:
    """Validates v2.0 access tokens issued by Entra ID for this API's app registration."""

    def __init__(self, tenant_id: str, audience: str, jwks_client: PyJWKClient | None = None):
        self._tenant_id = tenant_id
        self._audience = audience
        self._issuer = f"https://login.microsoftonline.com/{tenant_id}/v2.0"
        self._jwks_client = jwks_client or PyJWKClient(
            f"https://login.microsoftonline.com/{tenant_id}/discovery/v2.0/keys",
            cache_keys=True,
            lifespan=3600,
        )

    def validate(self, token: str) -> AuthenticatedUser:
        try:
            signing_key = self._jwks_client.get_signing_key_from_jwt(token)
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience=self._audience,
                issuer=self._issuer,
                leeway=60,
                options={"require": ["exp", "iat", "iss", "aud", "oid", "tid"]},
            )
        except jwt.PyJWTError as exc:
            raise AuthError(str(exc)) from exc
        if claims["tid"] != self._tenant_id:
            raise AuthError("token was issued for another tenant")
        return _user_from_claims(claims)


class DevTokenValidator:
    """HS256 tokens minted by `scripts/create_demo_users.py`. Refused in production by Settings."""

    def __init__(self, secret: str):
        self._secret = secret

    def validate(self, token: str) -> AuthenticatedUser:
        try:
            claims = jwt.decode(
                token,
                self._secret,
                algorithms=["HS256"],
                audience=DEV_AUDIENCE,
                issuer=DEV_ISSUER,
                options={"require": ["exp", "iat", "oid"]},
            )
        except jwt.PyJWTError as exc:
            raise AuthError(str(exc)) from exc
        return _user_from_claims(claims)


def create_dev_token(
    secret: str,
    *,
    object_id: str,
    name: str,
    groups: list[str],
    roles: list[str] | None = None,
    ttl_seconds: int = 8 * 3600,
) -> str:
    now = int(time.time())
    claims = {
        "iss": DEV_ISSUER,
        "aud": DEV_AUDIENCE,
        "iat": now,
        "exp": now + ttl_seconds,
        "oid": object_id,
        "name": name,
        "groups": groups,
        "roles": roles or [],
    }
    return jwt.encode(claims, secret, algorithm="HS256")


class GraphGroupResolver:
    """Fetches a user's security groups from Microsoft Graph (groups overage case).

    Needs an app registration with the `GroupMember.Read.All` application permission.
    """

    def __init__(self, tenant_id: str, client_id: str, client_secret: str, http: httpx.AsyncClient | None = None):
        self._tenant_id = tenant_id
        self._client_id = client_id
        self._client_secret = client_secret
        self._http = http or httpx.AsyncClient(timeout=10.0)
        self._token: str | None = None
        self._token_expires_at = 0.0

    async def _access_token(self) -> str:
        if self._token and time.monotonic() < self._token_expires_at - 60:
            return self._token
        response = await self._http.post(
            f"https://login.microsoftonline.com/{self._tenant_id}/oauth2/v2.0/token",
            data={
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "scope": "https://graph.microsoft.com/.default",
            },
        )
        response.raise_for_status()
        body = response.json()
        self._token = body["access_token"]
        self._token_expires_at = time.monotonic() + body["expires_in"]
        return self._token

    async def member_groups(self, object_id: str) -> tuple[str, ...]:
        token = await self._access_token()
        response = await self._http.post(
            f"https://graph.microsoft.com/v1.0/users/{quote(object_id, safe='')}/getMemberGroups",
            json={"securityEnabledOnly": True},
            headers={"Authorization": f"Bearer {token}"},
        )
        response.raise_for_status()
        return tuple(response.json()["value"])


def create_token_validator(settings: Settings) -> TokenValidator:
    if settings.auth_mode == "dev":
        assert settings.dev_jwt_secret is not None  # enforced by Settings
        return DevTokenValidator(settings.dev_jwt_secret.get_secret_value())
    if not (settings.entra_tenant_id and settings.entra_audience):
        raise ValueError("ENTRA_TENANT_ID and ENTRA_AUDIENCE are required when AUTH_MODE=entra")
    return EntraTokenValidator(settings.entra_tenant_id, settings.entra_audience)


def create_group_resolver(settings: Settings) -> GraphGroupResolver | None:
    secret = settings.graph_client_secret.get_secret_value() if settings.graph_client_secret else ""
    if settings.auth_mode != "entra" or not (settings.graph_client_id and secret):
        return None
    return GraphGroupResolver(settings.entra_tenant_id, settings.graph_client_id, secret)
