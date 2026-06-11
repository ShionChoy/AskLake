from __future__ import annotations

from pathlib import Path

import yaml

from engine.ports.auth import Principal

ANONYMOUS = Principal("anonymous", "public")


class StaticTokenAuthenticator:
    """Authenticator backed by a static token -> Principal map (config-driven, hermetic).

    Unknown/empty/None credential -> ANONYMOUS (role=public), never an error: missing
    credentials degrade to least privilege rather than 401."""

    def __init__(self, tokens: dict[str, Principal]):
        self._tokens = dict(tokens)

    @classmethod
    def from_yaml(cls, path: str | Path) -> StaticTokenAuthenticator:
        data = yaml.safe_load(Path(path).read_text()) or {}
        tokens = {
            tok: Principal(user=str(cfg.get("user", "")), role=str(cfg["role"]))
            for tok, cfg in (data.get("tokens", {}) or {}).items()
        }
        return cls(tokens)

    @property
    def roles(self) -> set[str]:
        return {p.role for p in self._tokens.values()}

    def authenticate(self, credential: str | None) -> Principal:
        if not credential:
            return ANONYMOUS
        return self._tokens.get(credential, ANONYMOUS)
