# tests/unit/test_api.py
from fastapi.testclient import TestClient

from api.main import create_app
from engine.lakehouse.duckdb_backend import DuckDBBackend

SEED_SQL = """
CREATE TABLE movies AS
SELECT * FROM (VALUES ('Inception', 2010, 8.8), ('Tenet', 2020, 7.3)) AS t(title, year, rating);
"""


def make_client():
    backend = DuckDBBackend()
    backend.setup(SEED_SQL)
    return TestClient(create_app(backend=backend))


def test_health_ok():
    r = make_client().get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_query_returns_rows():
    r = make_client().post("/query", json={"sql": "SELECT title FROM movies ORDER BY rating DESC"})
    assert r.status_code == 200
    body = r.json()
    assert body["columns"] == ["title"]
    assert body["rows"][0][0] == "Inception"


def test_query_error_returns_400():
    r = make_client().post("/query", json={"sql": "SELECT * FROM does_not_exist"})
    assert r.status_code == 400
    assert "error" in r.json()
