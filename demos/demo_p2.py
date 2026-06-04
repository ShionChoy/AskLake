"""Phase 2 demo: agentic self-correction + eval baseline-vs-agentic, hermetic (no API key).

Shows ONE live self-correction (bad `rating` column -> corrected `averageRating`) over a tiny
movies table with FakeLLMProvider, then prints the offline eval comparison."""

from __future__ import annotations

from engine.lakehouse.duckdb_backend import DuckDBBackend
from engine.llm.fake import FakeLLMProvider
from engine.retrieval.agentic_sql_path import AgenticSqlPath
from engine.semantic.raw_schema import RawSchemaProvider
from eval.hermetic import run_hermetic_comparison

SEED = (
    "CREATE TABLE movies AS SELECT * FROM "
    "(VALUES ('Alpha', 8.9), ('Beta', 7.0)) t(title, averageRating);"
)


def run_demo_p2() -> dict:
    backend = DuckDBBackend()
    backend.setup(SEED)
    llm = FakeLLMProvider(
        responses=[
            "SELECT title, rating FROM movies ORDER BY rating DESC",  # bad column
            "SELECT title, averageRating FROM movies ORDER BY averageRating DESC",  # corrected
        ]
    )
    rr = AgenticSqlPath(llm, RawSchemaProvider(backend), backend, max_retries=2).run(
        "highest rated movies"
    )
    baseline, agentic = run_hermetic_comparison()
    return {
        "sql": rr.sql,
        "rows": [list(r) for r in rr.result.rows] if rr.result else None,
        "narrative": rr.narrative,
        "chart_spec": rr.chart_spec,
        "baseline_execution_accuracy": baseline.execution_accuracy,
        "agentic_execution_accuracy": agentic.execution_accuracy,
    }


if __name__ == "__main__":
    out = run_demo_p2()
    print("corrected sql:", out["sql"])
    print("rows:", out["rows"])
    print("narrative:", out["narrative"])
    print("chart:", out["chart_spec"])
    print(f"baseline execution accuracy: {out['baseline_execution_accuracy']:.0%}")
    print(f"agentic execution accuracy:  {out['agentic_execution_accuracy']:.0%}")
    print("demo-p2 OK")
