import pytest
from fastapi.testclient import TestClient

from api.serve import build_app
from engine.graph.store import InMemoryGraphStore
from engine.lakehouse.duckdb_backend import DuckDBBackend
from engine.llm.fake import FakeLLMProvider
from engine.ports.graph_store import Triple


@pytest.fixture(autouse=True)
def _no_disk_graph(monkeypatch, tmp_path):
    # build_app reads the module global at call time; point it at a non-existent file so unit
    # tests never auto-load a real graph that may have been built locally.
    monkeypatch.setattr("api.serve.GRAPH_PATH", str(tmp_path / "no-graph.jsonl"))


def _graph_store() -> InMemoryGraphStore:
    s = InMemoryGraphStore()
    s.add(Triple("The Dark Knight", "DIRECTED_BY", "Christopher Nolan", "cmu:1"))
    s.add(Triple("The Dark Knight", "HAS_THEME", "identity", "cmu:1"))
    return s


def test_build_app_serves_grounded_trace():
    app = build_app(llm=FakeLLMProvider(["SELECT 1 AS x"]), backend=DuckDBBackend())
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    info = c.get("/info").json()
    assert info["model"] and info["provider"]
    out = c.post("/ask_trace", json={"question": "q"}).json()
    assert out["rows"] == [[1]]
    assert out["model"]
    assert any(s["step"].startswith("Generate SQL") for s in out["steps"])
    assert any(s["step"] == "Execute SQL" and s["ok"] for s in out["steps"])


def test_ask_trace_emits_ask_span_metric(monkeypatch):
    monkeypatch.setenv("ASKLAKE_OBSERVABILITY_BACKEND", "prometheus")
    app = build_app(llm=FakeLLMProvider(["SELECT 1 AS x"]), backend=DuckDBBackend())
    client = TestClient(app)
    client.post("/ask_trace", json={"question": "q"})
    registry = app.state.observability.registry
    assert registry.get_sample_value("asklake_spans_total", {"name": "ask"}) == 1.0


def test_endpoints_registered_without_boot_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ASKLAKE_LLM_PROVIDER", raising=False)
    app = build_app(backend=DuckDBBackend())  # no llm arg, no env -> no default provider
    c = TestClient(app)
    assert c.get("/info").status_code == 200
    out = c.post("/ask_trace", json={"question": "q"}).json()
    assert out["columns"] is None
    assert "sidebar" in out["narrative"].lower()  # friendly no-key payload, not a 500


def test_ask_trace_uses_per_request_credentials(monkeypatch):
    captured = {}

    def fake_make_provider(provider=None, api_key=None, model=None):
        captured["provider"] = provider
        captured["api_key"] = api_key
        captured["model"] = model
        return FakeLLMProvider(["SELECT 2 AS x"])

    monkeypatch.setattr("api.serve.make_provider", fake_make_provider)
    app = build_app(llm=FakeLLMProvider(["SELECT 1 AS x"]), backend=DuckDBBackend())
    c = TestClient(app)
    out = c.post(
        "/ask_trace",
        json={
            "question": "q",
            "provider": "deepseek",
            "model": "deepseek-v4-pro",
            "api_key": "sk-test",
        },
    ).json()
    assert captured == {"provider": "deepseek", "api_key": "sk-test", "model": "deepseek-v4-pro"}
    assert out["rows"] == [[2]]  # used the per-request provider, not the default (which returns 1)
    assert out["model"]


def test_ask_trace_redacts_api_key_in_errors(monkeypatch):
    def boom_make_provider(provider=None, api_key=None, model=None):
        raise RuntimeError(f"auth failed for key {api_key}")

    monkeypatch.setattr("api.serve.make_provider", boom_make_provider)
    app = build_app(llm=FakeLLMProvider(["SELECT 1 AS x"]), backend=DuckDBBackend())
    c = TestClient(app)
    out = c.post(
        "/ask_trace",
        json={"question": "q", "provider": "deepseek", "api_key": "sk-SECRET"},
    ).json()
    assert "sk-SECRET" not in out["narrative"]
    assert "***" in out["narrative"]


def test_ask_trace_no_typed_key_shows_sidebar_prompt(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ASKLAKE_LLM_PROVIDER", raising=False)
    app = build_app(backend=DuckDBBackend())  # no default provider
    c = TestClient(app)
    # The UI always sends `provider` from the selectbox even when no key is typed:
    out = c.post(
        "/ask_trace",
        json={"question": "q", "provider": "deepseek", "model": "deepseek-v4-flash"},
    ).json()
    assert "sidebar" in out["narrative"].lower()  # friendly prompt, not a 401


def test_ask_trace_blank_key_treated_as_no_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ASKLAKE_LLM_PROVIDER", raising=False)
    app = build_app(backend=DuckDBBackend())  # no default provider
    c = TestClient(app)
    out = c.post(
        "/ask_trace",
        json={"question": "q", "provider": "deepseek", "api_key": "   "},
    ).json()
    # A blank key -> friendly prompt, NOT an attempted provider call / 401.
    assert "sidebar" in out["narrative"].lower()


def test_ask_trace_graph_path_no_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ASKLAKE_LLM_PROVIDER", raising=False)
    app = build_app(backend=DuckDBBackend(), graph_store=_graph_store())  # no LLM key at all
    c = TestClient(app)
    out = c.post(
        "/ask_trace", json={"question": "themes of The Dark Knight", "path": "graph"}
    ).json()
    assert out["path"] == "graph"
    assert out["columns"] == ["subject", "relation", "object", "source"]
    assert any(row[2] == "identity" for row in out["rows"])  # object column carries the theme
    assert any(s["step"] == "Search knowledge graph" for s in out["steps"])


def test_ask_trace_fusion_merges_sql_and_graph():
    app = build_app(
        llm=FakeLLMProvider(["SELECT 1 AS x"]),
        backend=DuckDBBackend(),
        graph_store=_graph_store(),
    )
    c = TestClient(app)
    out = c.post(
        "/ask_trace", json={"question": "The Dark Knight rating and themes", "path": "fusion"}
    ).json()
    assert "sql" in out["path"] and "graph" in out["path"]  # "sql+graph"
    assert "[graph]" in out["narrative"]
    assert out["rows"] == [[1]]  # SQL table is the primary result


def test_ask_trace_sql_override_forces_sql():
    app = build_app(
        llm=FakeLLMProvider(["SELECT 7 AS x"]),
        backend=DuckDBBackend(),
        graph_store=_graph_store(),
    )
    c = TestClient(app)
    out = c.post("/ask_trace", json={"question": "common themes", "path": "sql"}).json()
    assert out["path"] == "sql"
    assert out["rows"] == [[7]]


def test_ask_trace_auto_routes_theme_question_to_graph():
    app = build_app(backend=DuckDBBackend(), graph_store=_graph_store())  # no key needed
    c = TestClient(app)
    out = c.post("/ask_trace", json={"question": "common themes in The Dark Knight"}).json()
    assert out["path"] == "graph"


def test_ask_trace_graph_requested_but_not_built():
    app = build_app(llm=FakeLLMProvider(["SELECT 1 AS x"]), backend=DuckDBBackend())  # no graph
    c = TestClient(app)
    out = c.post("/ask_trace", json={"question": "themes", "path": "graph"}).json()
    assert "build-graph" in out["narrative"]
    assert out["columns"] is None
    assert out["path"] == "graph"


def test_ask_trace_graph_includes_triples(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ASKLAKE_LLM_PROVIDER", raising=False)
    app = build_app(backend=DuckDBBackend(), graph_store=_graph_store())
    c = TestClient(app)
    out = c.post(
        "/ask_trace", json={"question": "themes of The Dark Knight", "path": "graph"}
    ).json()
    assert out["graph_triples"]
    assert all(len(t) == 4 for t in out["graph_triples"])
    assert any(t[0] == "The Dark Knight" for t in out["graph_triples"])


def test_ask_trace_fusion_includes_graph_triples():
    app = build_app(
        llm=FakeLLMProvider(["SELECT 1 AS x"]),
        backend=DuckDBBackend(),
        graph_store=_graph_store(),
    )
    c = TestClient(app)
    out = c.post(
        "/ask_trace", json={"question": "The Dark Knight rating and themes", "path": "fusion"}
    ).json()
    assert out["rows"] == [[1]]  # SQL table stays the primary result
    assert out["graph_triples"]  # graph triples ride alongside, not in rows
    assert all(len(t) == 4 for t in out["graph_triples"])
    assert any(t[2] == "identity" for t in out["graph_triples"])


def test_ask_trace_sql_only_has_no_graph_triples():
    app = build_app(
        llm=FakeLLMProvider(["SELECT 7 AS x"]),
        backend=DuckDBBackend(),
        graph_store=_graph_store(),
    )
    c = TestClient(app)
    out = c.post("/ask_trace", json={"question": "common themes", "path": "sql"}).json()
    assert out["path"] == "sql"
    assert out.get("graph_triples") is None


def test_serve_uses_grounded_path_by_default(monkeypatch):
    monkeypatch.delenv("ASKLAKE_AGENT", raising=False)
    import api.serve as serve
    from engine.lakehouse.duckdb_backend import DuckDBBackend
    from engine.llm.fake import FakeLLMProvider

    backend = DuckDBBackend()
    backend.setup("CREATE TABLE title_basics AS SELECT 1 AS tconst;")
    app = serve.build_app(llm=FakeLLMProvider(responses=["SELECT 1"]), backend=backend)
    assert app.state.sql_path_kind == "grounded"
