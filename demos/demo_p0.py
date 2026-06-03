# demos/demo_p0.py
"""Phase 0 demo: in-process query through the FastAPI app against a seed table.
No Docker, no LLM, no API key required."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import create_app
from engine.lakehouse.duckdb_backend import DuckDBBackend

SEED_SQL = """
CREATE TABLE movies AS
SELECT * FROM (VALUES
    ('Inception', 2010, 8.8),
    ('Interstellar', 2014, 8.7),
    ('Tenet', 2020, 7.3)
) AS t(title, year, rating);
"""

DEMO_SQL = "SELECT title, rating FROM movies WHERE year >= 2014 ORDER BY rating DESC"


def run_demo_p0() -> dict:
    backend = DuckDBBackend()
    backend.setup(SEED_SQL)
    client = TestClient(create_app(backend=backend))
    assert client.get("/health").json()["status"] == "ok"
    return client.post("/query", json={"sql": DEMO_SQL}).json()


if __name__ == "__main__":
    out = run_demo_p0()
    print("columns:", out["columns"])
    for row in out["rows"]:
        print(row)
    print("demo-p0 OK")
