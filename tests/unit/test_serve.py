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
