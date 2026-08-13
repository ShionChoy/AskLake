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
        f"  - {{token_sha256: {token_digest('tok_a')}, user: alice, role: analyst}}\n"
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
    bad.write_text("tokens:\n  tok: {user: x, role: superadmin}\n")
    monkeypatch.setattr("api.serve.AUTH_CONFIG", str(bad))
    with pytest.raises(ValueError, match="auth.yaml roles not in governance.yaml"):
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
