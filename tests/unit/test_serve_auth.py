import pytest
from fastapi.testclient import TestClient

from api.serve import build_app
from engine.lakehouse.duckdb_backend import DuckDBBackend
from engine.llm.fake import FakeLLMProvider


@pytest.fixture(autouse=True)
def _no_disk_graph(monkeypatch, tmp_path):
    monkeypatch.setattr("api.serve.GRAPH_PATH", str(tmp_path / "no-graph.jsonl"))


def _backend() -> DuckDBBackend:
    b = DuckDBBackend()
    b.setup(
        "CREATE TABLE title_ratings AS SELECT * FROM (VALUES "
        "('tt1', 9.0, 100000), ('tt2', 8.0, 10)) v(tconst, averageRating, numVotes);"
    )
    return b


def _auth_yaml(tmp_path):
    p = tmp_path / "auth.yaml"
    p.write_text("tokens:\n  tok_a: {user: alice, role: analyst}\n")
    return p


def test_public_request_sees_only_public_catalog(monkeypatch, tmp_path):
    monkeypatch.setattr("api.serve.AUTH_CONFIG", str(_auth_yaml(tmp_path)))
    app = build_app(llm=FakeLLMProvider(["SELECT tconst FROM title_ratings"]), backend=_backend())
    c = TestClient(app)
    out = c.post("/ask_trace", json={"question": "all films"}).json()  # no token -> public
    assert out["rows"] == [["tt1"]]  # tt2 (10 votes) hidden


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
