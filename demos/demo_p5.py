"""Phase 5 demo: Observability port filled (no-op -> Prometheus), hermetic (no API key,
no Docker, no live Prometheus).

Wraps the P2 self-correction scenario with decorator-adapters that emit to a
PrometheusObservability backed by an injected registry, then reports the collected
metrics + the Prometheus text exposition (what /metrics would serve)."""

from __future__ import annotations

from engine.lakehouse.duckdb_backend import DuckDBBackend
from engine.llm.fake import FakeLLMProvider
from engine.observability.instrumented import (
    ObservingLLMProvider,
    ObservingStorageBackend,
)
from engine.observability.prometheus import PrometheusObservability
from engine.retrieval.agentic_sql_path import AgenticSqlPath
from engine.semantic.raw_schema import RawSchemaProvider

SEED = (
    "CREATE TABLE movies AS SELECT * FROM "
    "(VALUES ('Alpha', 8.9), ('Beta', 7.0)) t(title, averageRating);"
)


def run_demo_p5() -> dict:
    obs = PrometheusObservability()
    backend = ObservingStorageBackend(DuckDBBackend(), obs)
    backend._inner.setup(SEED)  # seed the wrapped backend
    llm = ObservingLLMProvider(
        FakeLLMProvider(
            responses=[
                "SELECT title, rating FROM movies ORDER BY rating DESC",  # bad column
                "SELECT title, averageRating FROM movies ORDER BY averageRating DESC",  # fixed
            ]
        ),
        obs,
    )
    rr = AgenticSqlPath(llm, RawSchemaProvider(backend), backend, max_retries=2).run(
        "highest rated movies"
    )

    def sample(metric: str, name: str) -> float:
        return obs.registry.get_sample_value(metric, {"name": name}) or 0.0

    return {
        "sql": rr.sql,
        "rows": [list(r) for r in rr.result.rows] if rr.result else None,
        "llm_calls": sample("asklake_events_total", "llm_call"),
        "sql_errors": sample("asklake_events_total", "sql_error"),
        "storage_runs": sample("asklake_span_duration_seconds_count", "storage.run_sql"),
        "exposition": obs.exposition().decode(),
    }


if __name__ == "__main__":
    out = run_demo_p5()
    print("corrected sql:", out["sql"])
    print("rows:", out["rows"])
    print(f"llm calls: {out['llm_calls']:.0f}  (bad gen + corrected gen)")
    print(f"sql errors caught: {out['sql_errors']:.0f}  (the bad column)")
    print(f"storage runs: {out['storage_runs']:.0f}")
    print("--- /metrics exposition (excerpt) ---")
    for line in out["exposition"].splitlines():
        if line.startswith("asklake_") and "_total" in line:
            print(line)
    print("demo-p5 OK")
