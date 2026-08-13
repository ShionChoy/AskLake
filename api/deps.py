from __future__ import annotations

from fastapi import Header, HTTPException, Request, status

from engine.auth.static_token import AuthenticationError
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
    """Resolve a caller; anonymous access is explicit and invalid credentials return 401."""
    authenticator = getattr(request.app.state, "authenticator", None)
    if authenticator is None:
        return ANONYMOUS
    credential = _bearer(authorization)
    if authorization and credential is None:
        error = AuthenticationError("Authorization must use a Bearer credential")
    else:
        try:
            return authenticator.authenticate(credential)
        except AuthenticationError as exc:
            error = exc

    audit = getattr(request.app.state, "audit", None)
    if audit is not None:
        audit.write(
            event="authentication",
            decision="denied",
            reason_code=error.code,
            path=request.url.path,
            request_id=getattr(request.state, "request_id", ""),
        )
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="invalid or missing bearer credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
