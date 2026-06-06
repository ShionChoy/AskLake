from fastapi.testclient import TestClient

from api.serve import build_app
from engine.lakehouse.duckdb_backend import DuckDBBackend
from engine.llm.fake import FakeLLMProvider


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
            "model": "deepseek-reasoner",
            "api_key": "sk-test",
        },
    ).json()
    assert captured == {"provider": "deepseek", "api_key": "sk-test", "model": "deepseek-reasoner"}
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
        json={"question": "q", "provider": "deepseek", "model": "deepseek-chat"},
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
