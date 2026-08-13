from datetime import UTC, datetime, timedelta

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from engine.auth.oidc import OidcAuthenticator
from engine.auth.static_token import AuthenticationError
from engine.ports.auth import Principal


class _SigningKey:
    def __init__(self, key) -> None:
        self.key = key


class _JwksClient:
    def __init__(self, key) -> None:
        self.key = key

    def get_signing_key_from_jwt(self, _token: str) -> _SigningKey:
        return _SigningKey(self.key)


@pytest.fixture
def keys():
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private, private.public_key()


def _auth(public_key, **overrides) -> OidcAuthenticator:
    settings = {
        "issuer": "https://identity.example.com",
        "audience": "asklake-api",
        "jwks_url": "https://identity.example.com/.well-known/jwks.json",
        "role_mapping": {
            "asklake-public": "public",
            "asklake-analyst": "analyst",
            "asklake-steward": "steward",
        },
        "jwks_client": _JwksClient(public_key),
    }
    settings.update(overrides)
    return OidcAuthenticator(**settings)


def _token(private_key, **overrides) -> str:
    now = datetime.now(UTC)
    claims = {
        "iss": "https://identity.example.com",
        "aud": "asklake-api",
        "sub": "subject-123",
        "preferred_username": "alice@example.com",
        "groups": ["asklake-analyst"],
        "iat": now,
        "exp": now + timedelta(minutes=5),
    }
    claims.update(overrides)
    return jwt.encode(claims, private_key, algorithm="RS256", headers={"kid": "test-key"})


def test_valid_oidc_token_maps_external_group_to_internal_role(keys):
    private, public = keys
    assert _auth(public).authenticate(_token(private)) == Principal("alice@example.com", "analyst")


def test_oidc_rejects_wrong_audience_and_ambiguous_roles(keys):
    private, public = keys
    auth = _auth(public)
    with pytest.raises(AuthenticationError):
        auth.authenticate(_token(private, aud="another-api"))
    with pytest.raises(AuthenticationError, match="exactly one role"):
        auth.authenticate(_token(private, groups=["asklake-analyst", "asklake-steward"]))


def test_oidc_requires_bearer_by_default_but_can_use_explicit_anonymous_role(keys):
    _, public = keys
    with pytest.raises(AuthenticationError, match="required"):
        _auth(public).authenticate(None)
    anonymous = _auth(public, allow_anonymous=True)
    assert anonymous.authenticate(None) == Principal("anonymous", "public")


def test_oidc_rejects_insecure_endpoints_and_symmetric_algorithm(keys):
    _, public = keys
    with pytest.raises(ValueError, match="HTTPS"):
        _auth(public, issuer="http://identity.example.com")
    with pytest.raises(ValueError, match="asymmetric"):
        _auth(public, algorithms=("HS256",))
