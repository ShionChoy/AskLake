from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Principal:
    """An authenticated caller. `role` must be one of the dataset's governance roles."""

    user: str
    role: str
    credential_id: str = ""


@runtime_checkable
class Authenticator(Protocol):
    """Credential -> Principal. Invalid credentials raise; anonymous access is explicit."""

    def authenticate(self, credential: str | None) -> Principal: ...
