from __future__ import annotations

import getpass
import hashlib
import hmac
import re
from pathlib import Path
from typing import Any

import yaml

from engine.ports.auth import Principal

_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


class AuthenticationError(ValueError):
    def __init__(self, message: str, *, code: str = "invalid_credentials") -> None:
        super().__init__(message)
        self.code = code


def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class StaticTokenAuthenticator:
    """Hashed bearer-token authenticator for local or gateway-fronted deployments.

    Missing credentials may map to one explicitly configured anonymous role. Supplying an invalid
    or malformed credential never falls back to that role: it is an authentication failure. Raw
    tokens passed to the constructor are immediately hashed and are not retained.
    """

    def __init__(
        self,
        tokens: dict[str, Principal],
        *,
        hashed_tokens: dict[str, Principal] | None = None,
        allow_anonymous: bool = True,
        anonymous_role: str = "public",
    ) -> None:
        self._tokens = {token_digest(token): principal for token, principal in tokens.items()}
        for digest, principal in (hashed_tokens or {}).items():
            if not _SHA256.fullmatch(digest):
                raise AuthenticationError(
                    "configured token digest is not SHA-256", code="bad_config"
                )
            self._tokens[digest.lower()] = principal
        self._allow_anonymous = allow_anonymous
        self._anonymous = Principal("anonymous", anonymous_role)

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        anonymous_role: str | None = None,
    ) -> StaticTokenAuthenticator:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise AuthenticationError("authentication config must be a mapping", code="bad_config")
        allowed = {"version", "allow_anonymous", "anonymous_role", "credentials", "tokens"}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise AuthenticationError(
                f"unknown authentication config keys: {unknown}", code="bad_config"
            )
        version = int(raw.get("version", 1))
        if version not in {1, 2}:
            raise AuthenticationError(
                "unsupported authentication config version", code="bad_config"
            )
        if version >= 2 and raw.get("tokens"):
            raise AuthenticationError(
                "version-2 authentication config accepts only SHA-256 token digests",
                code="bad_config",
            )

        principals: dict[str, Principal] = {}
        credentials = raw.get("credentials", []) or []
        if not isinstance(credentials, list):
            raise AuthenticationError("credentials must be a list", code="bad_config")
        for index, item in enumerate(credentials):
            if not isinstance(item, dict):
                raise AuthenticationError(
                    f"credentials[{index}] must be a mapping", code="bad_config"
                )
            extra = sorted(set(item) - {"token_sha256", "user", "role"})
            if extra:
                raise AuthenticationError(
                    f"unknown credentials[{index}] keys: {extra}", code="bad_config"
                )
            digest = str(item.get("token_sha256", "")).lower()
            if not _SHA256.fullmatch(digest):
                raise AuthenticationError(
                    f"credentials[{index}].token_sha256 must be 64 hex characters",
                    code="bad_config",
                )
            if digest in principals:
                raise AuthenticationError("duplicate token digest", code="bad_config")
            principals[digest] = Principal(
                user=str(item.get("user", "")), role=str(item.get("role", ""))
            )

        # Version-1 compatibility for existing private configs. New/example configs never store
        # plaintext token keys.
        plaintext: dict[str, Principal] = {}
        tokens: Any = raw.get("tokens", {}) or {}
        if not isinstance(tokens, dict):
            raise AuthenticationError("tokens must be a mapping", code="bad_config")
        for token, cfg in tokens.items():
            if not isinstance(cfg, dict) or "role" not in cfg:
                raise AuthenticationError("each token must define a role", code="bad_config")
            plaintext[str(token)] = Principal(user=str(cfg.get("user", "")), role=str(cfg["role"]))

        configured_anonymous = anonymous_role or str(raw.get("anonymous_role", "public"))
        return cls(
            plaintext,
            hashed_tokens=principals,
            allow_anonymous=bool(raw.get("allow_anonymous", True)),
            anonymous_role=configured_anonymous,
        )

    @property
    def roles(self) -> set[str]:
        return {principal.role for principal in self._tokens.values()}

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
        digest = token_digest(credential)
        # compare_digest also makes the security intent explicit if the map implementation changes.
        for configured, principal in self._tokens.items():
            if hmac.compare_digest(digest, configured):
                return principal
        raise AuthenticationError("bearer credential is invalid")


def _main() -> None:
    token = getpass.getpass("Token to hash: ")
    if not token:
        raise SystemExit("token must not be empty")
    print(token_digest(token))


if __name__ == "__main__":
    _main()
