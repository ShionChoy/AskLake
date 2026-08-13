from __future__ import annotations

import argparse
import getpass
import hashlib
import hmac
import re
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from engine.ports.auth import Principal

_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class _Credential:
    principal: Principal
    credential_id: str = ""
    expires_at: datetime | None = None
    disabled: bool = False


def _expires_at(value: Any, where: str) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise AuthenticationError(
            f"{where} must be an ISO-8601 timestamp", code="bad_config"
        ) from exc
    if parsed.tzinfo is None:
        raise AuthenticationError(f"{where} must include a timezone", code="bad_config")
    return parsed.astimezone(UTC)


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

    method = "static_token"

    def __init__(
        self,
        tokens: dict[str, Principal],
        *,
        hashed_tokens: dict[str, Principal] | None = None,
        allow_anonymous: bool = True,
        anonymous_role: str = "public",
    ) -> None:
        self._tokens = {
            token_digest(token): _Credential(principal=principal)
            for token, principal in tokens.items()
        }
        for digest, principal in (hashed_tokens or {}).items():
            if not _SHA256.fullmatch(digest):
                raise AuthenticationError(
                    "configured token digest is not SHA-256", code="bad_config"
                )
            self._tokens[digest.lower()] = _Credential(principal=principal)
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
        version = int(raw.get("version", 0))
        if version != 2:
            raise AuthenticationError(
                "authentication config must use digest-only version 2", code="bad_config"
            )
        if "tokens" in raw:
            raise AuthenticationError(
                "version-2 authentication config accepts only SHA-256 token digests",
                code="bad_config",
            )
        allow_anonymous = raw.get("allow_anonymous", True)
        if not isinstance(allow_anonymous, bool):
            raise AuthenticationError("allow_anonymous must be true or false", code="bad_config")

        configured: dict[str, _Credential] = {}
        credential_ids: set[str] = set()
        credentials = raw.get("credentials", []) or []
        if not isinstance(credentials, list):
            raise AuthenticationError("credentials must be a list", code="bad_config")
        for index, item in enumerate(credentials):
            if not isinstance(item, dict):
                raise AuthenticationError(
                    f"credentials[{index}] must be a mapping", code="bad_config"
                )
            extra = sorted(
                set(item) - {"id", "token_sha256", "user", "role", "expires_at", "disabled"}
            )
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
            if digest in configured:
                raise AuthenticationError("duplicate token digest", code="bad_config")
            credential_id = str(item.get("id", "")).strip()
            user = str(item.get("user", "")).strip()
            role = str(item.get("role", "")).strip()
            if not credential_id:
                raise AuthenticationError(f"credentials[{index}].id is required", code="bad_config")
            if credential_id in credential_ids:
                raise AuthenticationError("duplicate credential id", code="bad_config")
            if not user or not role:
                raise AuthenticationError(
                    f"credentials[{index}] must define non-empty user and role",
                    code="bad_config",
                )
            credential_ids.add(credential_id)
            disabled = item.get("disabled", False)
            if not isinstance(disabled, bool):
                raise AuthenticationError(
                    f"credentials[{index}].disabled must be true or false", code="bad_config"
                )
            configured[digest] = _Credential(
                principal=Principal(user=user, role=role, credential_id=credential_id),
                credential_id=credential_id,
                expires_at=_expires_at(item.get("expires_at"), f"credentials[{index}].expires_at"),
                disabled=disabled,
            )

        configured_anonymous = anonymous_role or str(raw.get("anonymous_role", "public"))
        instance = cls(
            {},
            allow_anonymous=allow_anonymous,
            anonymous_role=configured_anonymous,
        )
        instance._tokens = configured
        now = datetime.now(UTC)
        has_active_credential = any(
            not item.disabled and (item.expires_at is None or item.expires_at > now)
            for item in configured.values()
        )
        if not instance._allow_anonymous and not has_active_credential:
            raise AuthenticationError(
                "authentication config has no active credentials and anonymous access is disabled",
                code="bad_config",
            )
        return instance

    @property
    def roles(self) -> set[str]:
        return {credential.principal.role for credential in self._tokens.values()}

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
        for configured, item in self._tokens.items():
            if hmac.compare_digest(digest, configured):
                if item.disabled:
                    raise AuthenticationError("bearer credential is disabled")
                if item.expires_at is not None and datetime.now(UTC) >= item.expires_at:
                    raise AuthenticationError(
                        "bearer credential has expired", code="credential_expired"
                    )
                return item.principal
        raise AuthenticationError("bearer credential is invalid")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate or hash an AskLake access token")
    parser.add_argument("command", nargs="?", choices=("hash", "generate"), default="hash")
    args = parser.parse_args()
    if args.command == "generate":
        token = secrets.token_urlsafe(32)
        print(f"token: {token}")
        print(f"token_sha256: {token_digest(token)}")
        return
    token = getpass.getpass("Token to hash: ")
    if not token:
        raise SystemExit("token must not be empty")
    print(token_digest(token))


if __name__ == "__main__":
    _main()
