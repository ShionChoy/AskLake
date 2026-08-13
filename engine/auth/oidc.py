from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlparse

import jwt
from jwt import PyJWKClient
from jwt.exceptions import PyJWKClientError, PyJWTError

from engine.auth.static_token import AuthenticationError
from engine.ports.auth import Principal

_ASYMMETRIC_ALGORITHMS = frozenset(
    {"RS256", "RS384", "RS512", "PS256", "PS384", "PS512", "ES256", "ES384", "ES512", "EdDSA"}
)


def _claim(claims: Mapping[str, Any], path: str) -> Any:
    value: Any = claims
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _https_url(value: str, where: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError(f"{where} must be an HTTPS URL without embedded credentials")
    return value


class OidcAuthenticator:
    """Validate OIDC access tokens against a pinned issuer, audience, JWKS, and role map.

    The caller cannot nominate an internal role. Exactly one externally asserted role must map to
    an AskLake role; missing or ambiguous mappings fail closed.
    """

    method = "oidc_jwt"

    def __init__(
        self,
        *,
        issuer: str,
        audience: str,
        jwks_url: str,
        role_mapping: Mapping[str, str],
        role_claim: str = "groups",
        user_claim: str = "preferred_username",
        algorithms: Sequence[str] = ("RS256",),
        allow_anonymous: bool = False,
        anonymous_role: str = "public",
        leeway_seconds: int = 30,
        jwks_client: PyJWKClient | None = None,
    ) -> None:
        self._issuer = _https_url(issuer, "OIDC issuer")
        self._audience = audience.strip()
        self._jwks_url = _https_url(jwks_url, "OIDC JWKS URL")
        self._role_mapping = {str(key): str(value) for key, value in role_mapping.items()}
        self._role_claim = role_claim.strip()
        self._user_claim = user_claim.strip()
        self._algorithms = tuple(str(item) for item in algorithms)
        self._allow_anonymous = allow_anonymous
        self._anonymous = Principal("anonymous", anonymous_role)
        self._leeway = int(leeway_seconds)
        if not self._audience or not self._role_mapping:
            raise ValueError("OIDC audience and role mapping are required")
        if not self._role_claim or not self._user_claim:
            raise ValueError("OIDC role and user claim names are required")
        if not self._algorithms or not set(self._algorithms) <= _ASYMMETRIC_ALGORITHMS:
            raise ValueError("OIDC algorithms must explicitly allow only asymmetric signatures")
        if self._leeway < 0 or self._leeway > 300:
            raise ValueError("OIDC clock leeway must be between 0 and 300 seconds")
        self._jwks = jwks_client or PyJWKClient(
            self._jwks_url,
            cache_keys=True,
            lifespan=300,
            timeout=5,
        )

    @property
    def roles(self) -> set[str]:
        return set(self._role_mapping.values())

    @property
    def anonymous_role(self) -> str | None:
        return self._anonymous.role if self._allow_anonymous else None

    def authenticate(self, credential: str | None) -> Principal:
        if not credential:
            if self._allow_anonymous:
                return self._anonymous
            raise AuthenticationError(
                "bearer credentials are required", code="credentials_required"
            )
        try:
            signing_key = self._jwks.get_signing_key_from_jwt(credential)
            claims = jwt.decode(
                credential,
                signing_key.key,
                algorithms=list(self._algorithms),
                audience=self._audience,
                issuer=self._issuer,
                leeway=self._leeway,
                options={"require": ["exp", "iat", "iss", "sub", "aud"]},
            )
        except (PyJWTError, PyJWKClientError, ValueError) as exc:
            raise AuthenticationError("OIDC bearer credential is invalid") from exc

        asserted = _claim(claims, self._role_claim)
        external_roles = [asserted] if isinstance(asserted, str) else asserted
        if not isinstance(external_roles, (list, tuple, set)):
            raise AuthenticationError("OIDC role claim is missing", code="role_claim_missing")
        mapped = {
            self._role_mapping[str(external)]
            for external in external_roles
            if str(external) in self._role_mapping
        }
        if len(mapped) != 1:
            raise AuthenticationError(
                "OIDC credential must resolve to exactly one role", code="role_mapping_denied"
            )
        user = _claim(claims, self._user_claim) or claims.get("sub")
        if not isinstance(user, str) or not user.strip():
            raise AuthenticationError("OIDC user claim is missing", code="user_claim_missing")
        return Principal(
            user=user.strip(),
            role=mapped.pop(),
            credential_id=str(claims.get("jti", "")),
        )
