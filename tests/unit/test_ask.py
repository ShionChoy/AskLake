from fastapi.testclient import TestClient

from api.main import create_app
from engine.lakehouse.duckdb_backend import DuckDBBackend
from engine.llm.fake import FakeLLMProvider
from engine.retrieval.sql_path import SqlPath
from engine.semantic.raw_schema import RawSchemaProvider

SEED = "CREATE TABLE m AS SELECT * FROM (VALUES ('A', 8.8), ('B', 7.0)) t(title, rating);"


def test_ask_returns_sql_table_and_chart():
    backend = DuckDBBackend()
    backend.setup(SEED)
    sql = "SELECT title, rating FROM m ORDER BY rating DESC"
    sql_path = SqlPath(FakeLLMProvider(responses=[sql]), RawSchemaProvider(backend), backend)
    client = TestClient(create_app(backend=backend, sql_path=sql_path))

    r = client.post("/ask", json={"question": "top movies?"})
    assert r.status_code == 200
    body = r.json()
    assert body["sql"].startswith("SELECT")
    assert body["columns"] == ["title", "rating"]
    assert body["rows"][0][0] == "A"
    assert body["chart_spec"]["type"] == "bar"
    assert body["path"] == "sql"
