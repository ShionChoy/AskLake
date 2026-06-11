from __future__ import annotations

from fastapi import Header, Request

from engine.ports.auth import Principal

ANONYMOUS = Principal("anonymous", "public")


def _bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip() or None
    return None


def require_principal(
    request: Request, authorization: str | None = Header(default=None)
) -> Principal:
    """Resolve the caller's Principal from `Authorization: Bearer <token>`.

    Reads `request.app.state.authenticator`. Missing/invalid token -> anonymous/public."""
    authenticator = getattr(request.app.state, "authenticator", None)
    if authenticator is None:
        return ANONYMOUS
    return authenticator.authenticate(_bearer(authorization))
