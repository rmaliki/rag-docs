import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.auth.entra import DEV_AUDIENCE, DEV_ISSUER, AuthError, EntraTokenValidator, create_dev_token
from app.auth.permissions import resolve_permissions
from app.config import Settings
from app.generation.rag_service import RagAnswer
from app.main import create_app
from tests.conftest import DEV_SECRET

TENANT = "11111111-1111-4111-8111-111111111111"
AUDIENCE = "22222222-2222-4222-8222-222222222222"

# ------------------------------------------------------- Entra token validation


@pytest.fixture(scope="module")
def signing_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


class StaticJwks:
    """Stands in for PyJWKClient: always returns the test public key."""

    def __init__(self, public_key):
        self.key = public_key

    def get_signing_key_from_jwt(self, token):
        return self


@pytest.fixture
def validator(signing_key):
    return EntraTokenValidator(TENANT, AUDIENCE, jwks_client=StaticJwks(signing_key.public_key()))


def entra_token(key, algorithm="RS256", **overrides) -> str:
    now = int(time.time())
    claims = {
        "iss": f"https://login.microsoftonline.com/{TENANT}/v2.0",
        "aud": AUDIENCE,
        "tid": TENANT,
        "oid": "user-1",
        "iat": now,
        "exp": now + 600,
        "groups": ["hr-team-id", "all-employees-id"],
        "roles": ["Documents.Admin"],
    }
    claims.update(overrides)
    return jwt.encode({k: v for k, v in claims.items() if v is not None}, key, algorithm=algorithm)


def test_valid_entra_token(validator, signing_key):
    user = validator.validate(entra_token(signing_key))
    assert user.object_id == "user-1"
    assert user.groups == ("hr-team-id", "all-employees-id")
    assert user.roles == ("Documents.Admin",)
    assert not user.groups_overage


@pytest.mark.parametrize(
    "overrides",
    [
        {"aud": "another-api"},
        {"iss": "https://login.microsoftonline.com/another-tenant/v2.0"},
        {"tid": "33333333-3333-4333-8333-333333333333"},
        {"exp": int(time.time()) - 3600},
        {"oid": None},
    ],
    ids=["wrong-audience", "wrong-issuer", "wrong-tenant", "expired", "missing-oid"],
)
def test_invalid_entra_tokens_are_rejected(validator, signing_key, overrides):
    with pytest.raises(AuthError):
        validator.validate(entra_token(signing_key, **overrides))


def test_token_signed_by_another_key_is_rejected(validator):
    attacker_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    with pytest.raises(AuthError):
        validator.validate(entra_token(attacker_key))


def test_unsigned_token_is_rejected(validator):
    with pytest.raises(AuthError):
        validator.validate(entra_token(None, algorithm="none"))


def test_groups_overage_fails_closed(validator, signing_key):
    token = entra_token(signing_key, groups=None, _claim_names={"groups": "src1"})
    user = validator.validate(token)
    assert user.groups_overage
    with pytest.raises(ValueError):
        resolve_permissions(user)


def test_permissions_are_derived_from_the_token(validator, signing_key):
    permissions = resolve_permissions(validator.validate(entra_token(signing_key)))
    assert permissions.principals == {"user:user-1", "group:hr-team-id", "group:all-employees-id"}
    assert permissions.has_role("Documents.Admin")


def test_dev_auth_is_refused_in_production():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, environment="production", auth_mode="dev", dev_jwt_secret=DEV_SECRET)


def test_dev_auth_needs_a_strong_secret():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, environment="development", auth_mode="dev", dev_jwt_secret="short")


# ---------------------------------------------------------------------- API


class RecordingRagService:
    def __init__(self):
        self.calls = []

    async def answer_question(self, question, permissions):
        self.calls.append((question, permissions))
        return RagAnswer(answer="ok", citations=[], insufficient_evidence=False)


@pytest.fixture
def api(settings):
    service = RecordingRagService()
    return TestClient(create_app(settings, rag_service=service)), service


def auth_header(groups=("marketing-team",), roles=()) -> dict[str, str]:
    token = create_dev_token(
        DEV_SECRET, object_id="u-1", name="Test User", groups=list(groups), roles=list(roles)
    )
    return {"Authorization": f"Bearer {token}"}


def test_chat_requires_a_token(api):
    client, service = api
    response = client.post("/chat", json={"question": "What is our Q3 budget?"})
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert service.calls == []


def test_chat_rejects_an_invalid_token(api):
    client, _ = api
    response = client.post("/chat", json={"question": "hi"}, headers={"Authorization": "Bearer not-a-jwt"})
    assert response.status_code == 401


def test_chat_rejects_client_supplied_groups(api):
    client, service = api
    response = client.post(
        "/chat", json={"question": "What are the executive salaries?", "groups": ["hr-team"]}, headers=auth_header()
    )
    assert response.status_code == 422
    assert service.calls == []


def test_chat_uses_groups_from_the_token(api):
    client, service = api
    body = {"question": "  What is our latest campaign strategy?  "}
    response = client.post("/chat", json=body, headers=auth_header())
    assert response.status_code == 200
    question, permissions = service.calls[0]
    assert question == "What is our latest campaign strategy?"
    assert permissions.principals == {"user:u-1", "group:marketing-team"}


def test_group_overage_without_graph_resolver_is_forbidden(api):
    client, service = api
    now = int(time.time())
    token = jwt.encode(
        {"iss": DEV_ISSUER, "aud": DEV_AUDIENCE, "iat": now, "exp": now + 600, "oid": "u-2",
         "_claim_names": {"groups": "src1"}},
        DEV_SECRET,
        algorithm="HS256",
    )
    response = client.post("/chat", json={"question": "hi"}, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 403
    assert service.calls == []


def test_upload_requires_the_admin_role(api):
    client, _ = api
    response = client.post(
        "/documents",
        headers=auth_header(groups=("hr-team",)),
        files={"file": ("policy.md", b"# Policy", "text/markdown")},
        data={"metadata": '{"document_id": "doc-9", "title": "T", "department": "HR", "allowed_groups": ["hr-team"]}'},
    )
    assert response.status_code == 403


def test_liveness_needs_no_token(api):
    client, _ = api
    assert client.get("/health/live").json() == {"status": "ok"}
