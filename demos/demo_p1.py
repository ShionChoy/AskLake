"""Phase 1 demo: NL -> SQL -> result -> chart, end-to-end and hermetic.
Builds a tiny IMDb-shaped parquet fixture and uses FakeLLMProvider (canned SQL)
so the demo runs in CI with no API key. The interactive demo uses AnthropicProvider."""

from __future__ import annotations

import tempfile

import duckdb
from fastapi.testclient import TestClient

from api.main import create_app
from engine.lakehouse.duckdb_backend import DuckDBBackend
from engine.llm.fake import FakeLLMProvider
from engine.retrieval.sql_path import SqlPath
from engine.semantic.raw_schema import RawSchemaProvider

CANNED_SQL = (
    "SELECT b.primaryTitle AS title, r.averageRating AS rating "
    "FROM title_basics b JOIN title_ratings r ON b.tconst = r.tconst "
    "WHERE b.startYear > 2010 AND b.genres LIKE '%Sci-Fi%' "
    "ORDER BY r.averageRating DESC LIMIT 10"
)


def _build_fixture(out_dir: str) -> None:
    con = duckdb.connect()
    con.execute(
        """
        CREATE TABLE title_basics AS SELECT * FROM (VALUES
            ('tt1', 'Sci A', 2014, 'Sci-Fi'),
            ('tt2', 'Sci B', 2012, 'Sci-Fi,Action'),
            ('tt3', 'Old Sci', 2001, 'Sci-Fi')
        ) t(tconst, primaryTitle, startYear, genres);
        CREATE TABLE title_ratings AS SELECT * FROM (VALUES
            ('tt1', 8.9, 90000), ('tt2', 7.5, 40000), ('tt3', 8.0, 10000)
        ) t(tconst, averageRating, numVotes);
        """
    )
    con.execute(f"COPY title_basics TO '{out_dir}/title_basics.parquet' (FORMAT PARQUET)")
    con.execute(f"COPY title_ratings TO '{out_dir}/title_ratings.parquet' (FORMAT PARQUET)")


def run_demo_p1() -> dict:
    tmp = tempfile.mkdtemp()
    _build_fixture(tmp)
    backend = DuckDBBackend(parquet_dir=tmp)
    sql_path = SqlPath(FakeLLMProvider(responses=[CANNED_SQL]), RawSchemaProvider(backend), backend)
    client = TestClient(create_app(backend=backend, sql_path=sql_path))
    return client.post("/ask", json={"question": "highest-rated sci-fi after 2010, top 10"}).json()


if __name__ == "__main__":
    out = run_demo_p1()
    print("sql:", out["sql"])
    print("columns:", out["columns"])
    for row in out["rows"]:
        print(row)
    print("chart:", out["chart_spec"])
    print("demo-p1 OK")
