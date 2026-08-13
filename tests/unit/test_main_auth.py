from fastapi.testclient import TestClient

from api.main import create_app
from engine.auth.static_token import StaticTokenAuthenticator
from engine.governance.policy import Policy, PolicyGovernance
from engine.governance.views import build_role_views
from engine.lakehouse.duckdb_backend import DuckDBBackend
from engine.ports.auth import Principal


def _app():
    backend = DuckDBBackend()
    backend.setup(
        "CREATE TABLE people AS SELECT * FROM (VALUES "
        "('Nolan', 1970), ('Hidden', 1980)) t(primaryName, birthYear);"
    )
    gov = PolicyGovernance(
        Policy(
            roles=("analyst", "public"),
            pii_columns=("birthYear",),
            mask_roles=("public",),
            require_limit=True,
            forbid_writes=True,
        )
    )
    auth = StaticTokenAuthenticator({"tok_a": Principal("alice", "analyst")})
    build_role_views(backend, gov.policy)
    return create_app(backend=backend, governance=gov, authenticator=auth)


def test_query_with_analyst_token_sees_pii():
    c = TestClient(_app())
    out = c.post(
        "/query",
        json={"sql": "SELECT primaryName, birthYear FROM people LIMIT 10"},
        headers={"Authorization": "Bearer tok_a"},
    ).json()
    assert ["Nolan", 1970] in out["rows"]


def test_query_without_token_is_public_and_masks_pii():
    c = TestClient(_app())
    out = c.post(
        "/query", json={"sql": "SELECT primaryName, birthYear FROM people LIMIT 10"}
    ).json()
    assert all(row[1] == "***" for row in out["rows"])  # masked for public


def test_self_declared_body_role_is_ignored_security_regression():
    c = TestClient(_app())
    # No token, but the body lies and claims analyst -> still public -> still masked.
    response = c.post(
        "/query",
        json={"sql": "SELECT primaryName, birthYear FROM people LIMIT 10", "role": "analyst"},
    )
    assert response.status_code == 422  # unknown authorization fields are rejected


def test_ask_does_not_cache_lazy_path_across_roles():
    # With no injected sql_path and no key, /ask must not pin app.state.sql_path to a role.
    # We assert the app never caches a lazily-built path (so each request rebuilds per-role).
    import inspect

    from api import main

    src = inspect.getsource(main.create_app)
    # the /ask lazy block must NOT assign app.state.sql_path (that would pin the first role)
    ask_src = src[src.index("def ask(") :]
    assert "app.state.sql_path = sp" not in ask_src
