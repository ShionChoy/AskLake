from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from api.deps import require_principal
from engine.auth.static_token import StaticTokenAuthenticator
from engine.ports.auth import Principal


def _app() -> FastAPI:
    app = FastAPI()
    app.state.authenticator = StaticTokenAuthenticator({"tok_a": Principal("alice", "analyst")})

    @app.get("/whoami")
    def whoami(principal: Principal = Depends(require_principal)) -> dict:  # noqa: B008
        return {"user": principal.user, "role": principal.role}

    return app


def test_valid_bearer_returns_principal():
    c = TestClient(_app())
    out = c.get("/whoami", headers={"Authorization": "Bearer tok_a"}).json()
    assert out == {"user": "alice", "role": "analyst"}


def test_no_header_degrades_to_public():
    out = TestClient(_app()).get("/whoami").json()
    assert out == {"user": "anonymous", "role": "public"}


def test_unknown_token_degrades_to_public():
    c = TestClient(_app())
    out = c.get("/whoami", headers={"Authorization": "Bearer nope"}).json()
    assert out == {"user": "anonymous", "role": "public"}


def test_non_bearer_header_degrades_to_public():
    c = TestClient(_app())
    out = c.get("/whoami", headers={"Authorization": "Basic tok_a"}).json()
    assert out == {"user": "anonymous", "role": "public"}
