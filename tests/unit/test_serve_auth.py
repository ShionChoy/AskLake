import json

import pytest
from fastapi.testclient import TestClient

from api.serve import build_app
from engine.auth.static_token import token_digest
from engine.lakehouse.duckdb_backend import DuckDBBackend
from engine.llm.fake import FakeLLMProvider


@pytest.fixture(autouse=True)
def _no_disk_graph(monkeypatch, tmp_path):
    monkeypatch.setattr("api.serve.GRAPH_PATH", str(tmp_path / "no-graph.jsonl"))


def _backend() -> DuckDBBackend:
    b = DuckDBBackend()
    b.setup(
        "CREATE TABLE title_basics AS SELECT * FROM (VALUES "
        "('tt1', false), ('tt2', true)) v(tconst, isAdult);"
        "CREATE TABLE title_ratings AS SELECT * FROM (VALUES "
        "('tt1', 9.0, 100000), ('tt2', 8.0, 10)) v(tconst, averageRating, numVotes);"
    )
    return b


def _auth_yaml(tmp_path):
    p = tmp_path / "auth.yaml"
    p.write_text(
        "version: 2\ncredentials:\n"
        f"  - {{id: analyst-primary, token_sha256: {token_digest('tok_a')}, "
        "user: alice, role: analyst}\n"
        f"  - {{id: steward-primary, token_sha256: {token_digest('tok_s')}, "
        "user: sam, role: steward}\n"
    )
    return p


def test_public_request_excludes_adult_content(monkeypatch, tmp_path):
    monkeypatch.setattr("api.serve.AUTH_CONFIG", str(_auth_yaml(tmp_path)))
    app = build_app(llm=FakeLLMProvider(["SELECT tconst FROM title_ratings"]), backend=_backend())
    c = TestClient(app)
    out = c.post("/ask_trace", json={"question": "all films"}).json()  # no token -> public
    assert out["rows"] == [["tt1"]]  # tt2 is adult; vote count is not an authorization rule


def test_analyst_token_sees_full_catalog(monkeypatch, tmp_path):
    monkeypatch.setattr("api.serve.AUTH_CONFIG", str(_auth_yaml(tmp_path)))
    app = build_app(llm=FakeLLMProvider(["SELECT tconst FROM title_ratings"]), backend=_backend())
    c = TestClient(app)
    out = c.post(
        "/ask_trace", json={"question": "all films"}, headers={"Authorization": "Bearer tok_a"}
    ).json()
    assert {r[0] for r in out["rows"]} == {"tt1", "tt2"}


def test_build_app_raises_when_auth_role_not_in_governance(monkeypatch, tmp_path):
    bad = tmp_path / "auth.yaml"
    bad.write_text(
        "version: 2\ncredentials:\n"
        f"  - {{id: bad-role, token_sha256: {token_digest('tok')}, "
        "user: x, role: superadmin}\n"
    )
    monkeypatch.setattr("api.serve.AUTH_CONFIG", str(bad))
    with pytest.raises(ValueError, match="authentication roles not in governance.yaml"):
        build_app(backend=_backend())


def test_public_cannot_use_raw_sql_but_analyst_can(monkeypatch, tmp_path):
    monkeypatch.setattr("api.serve.AUTH_CONFIG", str(_auth_yaml(tmp_path)))
    app = build_app(backend=_backend())
    client = TestClient(app)
    public = client.post("/query", json={"sql": "SELECT tconst FROM title_ratings"})
    assert public.status_code == 403
    assert public.json()["code"] == "action_denied"
    analyst = client.post(
        "/query",
        json={"sql": "SELECT tconst FROM title_ratings"},
        headers={"Authorization": "Bearer tok_a"},
    )
    assert analyst.status_code == 200
    assert {row[0] for row in analyst.json()["rows"]} == {"tt1", "tt2"}


def test_raw_sql_cannot_bypass_role_schema(monkeypatch, tmp_path):
    monkeypatch.setattr("api.serve.AUTH_CONFIG", str(_auth_yaml(tmp_path)))
    client = TestClient(build_app(backend=_backend()))
    response = client.post(
        "/query",
        json={"sql": "SELECT * FROM main.title_ratings"},
        headers={"Authorization": "Bearer tok_a"},
    )
    assert response.status_code == 403
    assert response.json()["code"] == "schema_access_denied"


def test_invalid_token_is_not_downgraded_to_public(monkeypatch, tmp_path):
    monkeypatch.setattr("api.serve.AUTH_CONFIG", str(_auth_yaml(tmp_path)))
    client = TestClient(build_app(backend=_backend()))
    response = client.post(
        "/ask_trace",
        json={"question": "all films"},
        headers={"Authorization": "Bearer invalid"},
    )
    assert response.status_code == 401


def test_session_exposes_effective_identity_and_capabilities(monkeypatch, tmp_path):
    monkeypatch.setattr("api.serve.AUTH_CONFIG", str(_auth_yaml(tmp_path)))
    client = TestClient(build_app(backend=_backend()))

    public = client.get("/session")
    assert public.status_code == 200
    assert public.headers["cache-control"] == "no-store"
    assert public.json()["principal"] == {
        "user": "anonymous",
        "role": "public",
        "authenticated": False,
        "authentication_method": "static_token",
        "credential_id": "",
    }
    assert "raw_sql" not in public.json()["governance"]["actions"]
    assert public.json()["governance"]["row_filtered_tables"]

    analyst = client.get("/session", headers={"Authorization": "Bearer tok_a"})
    assert analyst.json()["principal"] == {
        "user": "alice",
        "role": "analyst",
        "authenticated": True,
        "authentication_method": "static_token",
        "credential_id": "analyst-primary",
    }
    assert "raw_sql" in analyst.json()["governance"]["actions"]


def test_only_steward_can_export_bounded_formula_safe_csv(monkeypatch, tmp_path):
    monkeypatch.setattr("api.serve.AUTH_CONFIG", str(_auth_yaml(tmp_path)))
    client = TestClient(build_app(backend=_backend()))
    payload = {"sql": "SELECT '=1+1' AS title"}

    public = client.post("/export", json=payload)
    assert public.status_code == 403
    analyst = client.post("/export", json=payload, headers={"Authorization": "Bearer tok_a"})
    assert analyst.status_code == 403

    steward = client.post("/export", json=payload, headers={"Authorization": "Bearer tok_s"})
    assert steward.status_code == 200
    assert steward.headers["x-asklake-role"] == "steward"
    assert steward.headers["x-asklake-row-count"] == "1"
    assert steward.headers["x-content-type-options"] == "nosniff"
    assert "'=1+1" in steward.content.decode("utf-8-sig")


def test_oidc_mode_is_fail_closed_without_a_bearer_token(monkeypatch):
    monkeypatch.setattr("api.serve.AUTH_MODE", "oidc")
    monkeypatch.setattr("api.serve.OIDC_ISSUER", "https://identity.example.com")
    monkeypatch.setattr("api.serve.OIDC_AUDIENCE", "asklake-api")
    monkeypatch.setattr(
        "api.serve.OIDC_JWKS_URL", "https://identity.example.com/.well-known/jwks.json"
    )
    monkeypatch.setattr("api.serve.OIDC_ROLE_MAPPING", '{"asklake-analyst":"analyst"}')
    monkeypatch.setattr("api.serve.OIDC_ALLOW_ANONYMOUS", "false")
    client = TestClient(build_app(backend=_backend()))
    assert client.get("/session").status_code == 401


def test_audit_event_carries_rotation_credential_id(monkeypatch, tmp_path):
    audit_path = tmp_path / "events.jsonl"
    monkeypatch.setattr("api.serve.AUTH_CONFIG", str(_auth_yaml(tmp_path)))
    monkeypatch.setattr("api.serve.AUDIT_PATH", str(audit_path))
    client = TestClient(build_app(backend=_backend()))
    response = client.post(
        "/query",
        json={"sql": "SELECT 1 AS ok"},
        headers={"Authorization": "Bearer tok_a"},
    )
    assert response.status_code == 200
    event = json.loads(audit_path.read_text().splitlines()[-1])
    assert event["credential_id"] == "analyst-primary"
