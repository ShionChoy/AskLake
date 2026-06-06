"""Production entrypoint: the grounded, self-correcting NL->SQL app over real data.

Builds the FastAPI app with a live LLM (DeepSeek by default; Anthropic supported), the
semantic layer, agentic self-correction, and DuckDB memory guardrails — plus a per-request
processing trace exposed at /info and /ask_trace (consumed by the Streamlit UI).

Run:
    DEEPSEEK_API_KEY=... uv run uvicorn api.serve:build_app --factory
    # or: DEEPSEEK_API_KEY=... uv run python -m api.serve

The hermetic test app remains `api.main:create_app` (needs no key or data)."""

from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi import FastAPI

from api.main import create_app
from engine.lakehouse.duckdb_backend import DuckDBBackend
from engine.llm.factory import make_provider
from engine.ports.llm import LLMProvider
from engine.ports.storage import QueryResult, StorageBackend
from engine.retrieval.agentic_sql_path import AgenticSqlPath
from engine.semantic.semantic_layer import SemanticLayerProvider

PARQUET_DIR = os.environ.get("ASKLAKE_PARQUET_DIR", "data/imdb/parquet")
SEMANTIC_YAML = "datasets/imdb_cmu/semantic.yaml"


def apply_duckdb_guardrails(
    backend: StorageBackend,
    memory_limit: str = "2GB",
    max_temp_size: str = "4GB",
    threads: int = 2,
) -> None:
    """Cap RAM + on-disk spill so a runaway query fails fast (catchable) instead of OOM."""
    for stmt in (
        f"SET memory_limit='{memory_limit}'",
        f"SET max_temp_directory_size='{max_temp_size}'",
        f"SET threads={threads}",
    ):
        try:
            backend.run_sql(stmt)
        except Exception:  # noqa: BLE001
            pass


class _TraceLog:
    """Ordered, request-scoped record of backend processing steps (for the UI)."""

    def __init__(self) -> None:
        self.steps: list[dict] = []

    def reset(self) -> None:
        self.steps = []

    def add(
        self,
        step: str,
        *,
        ok: bool = True,
        ms: float = 0.0,
        detail: str = "",
        sql: str | None = None,
    ) -> None:
        self.steps.append(
            {"step": step, "ok": ok, "ms": round(ms, 1), "detail": detail, "sql": sql}
        )


class _TracingSchema:
    def __init__(self, inner, log: _TraceLog) -> None:
        self._inner = inner
        self._log = log

    def schema_context(self, question: str) -> str:
        t0 = time.perf_counter()
        ctx = self._inner.schema_context(question)
        self._log.add(
            "Retrieve semantic schema context",
            ms=(time.perf_counter() - t0) * 1000,
            detail=f"grounded the LLM with {len(ctx)} chars of curated schema "
            "(tables, columns, synonyms, few-shots)",
        )
        return ctx


class _TracingLLM:
    def __init__(self, inner, log: _TraceLog) -> None:
        self._inner = inner
        self._log = log

    def complete(self, prompt: str, system: str | None = None) -> str:
        t0 = time.perf_counter()
        out = self._inner.complete(prompt, system=system)
        self._log.add(
            "Generate SQL (LLM)",
            ms=(time.perf_counter() - t0) * 1000,
            detail="model produced a candidate query",
            sql=out.strip(),
        )
        return out


class _TracingBackend:
    def __init__(self, inner, log: _TraceLog) -> None:
        self._inner = inner
        self._log = log

    def run_sql(self, sql: str) -> QueryResult:
        t0 = time.perf_counter()
        try:
            r = self._inner.run_sql(sql)
            self._log.add(
                "Execute SQL",
                ok=True,
                ms=(time.perf_counter() - t0) * 1000,
                detail=f"{len(r.rows)} row(s) returned",
                sql=sql,
            )
            return r
        except Exception as exc:  # noqa: BLE001
            self._log.add(
                "Execute SQL",
                ok=False,
                ms=(time.perf_counter() - t0) * 1000,
                detail=f"error: {exc}",
                sql=sql,
            )
            raise

    def list_tables(self):
        return self._inner.list_tables()


def build_app(llm: LLMProvider | None = None, backend: StorageBackend | None = None) -> FastAPI:
    """Build the production app. With a configured provider it serves the grounded + agentic
    /ask plus /info and /ask_trace; with no provider it still boots (/health, /query work)."""
    if backend is None:
        parquet = PARQUET_DIR if Path(PARQUET_DIR).exists() else None
        backend = DuckDBBackend(parquet_dir=parquet)
        apply_duckdb_guardrails(backend)
    if llm is None:
        try:
            llm = make_provider()
        except Exception as exc:  # noqa: BLE001
            print(f"[api.serve] no LLM provider configured ({exc}); /ask disabled, /query works")
            return create_app(backend=backend)

    model_name = getattr(llm, "_model", None) or type(llm).__name__
    provider_name = type(llm).__name__
    log = _TraceLog()
    tbackend = _TracingBackend(backend, log)
    tllm = _TracingLLM(llm, log)
    tschema = _TracingSchema(SemanticLayerProvider.from_yaml(SEMANTIC_YAML), log)
    path = AgenticSqlPath(tllm, tschema, tbackend, max_retries=2)
    app = create_app(backend=tbackend, sql_path=path)
    app.state.trace_log = log
    app.state.model_name = model_name
    app.state.provider_name = provider_name

    @app.get("/info")
    def info() -> dict:
        return {
            "provider": provider_name,
            "model": model_name,
            "path": "semantic-grounded + self-correcting (agentic)",
        }

    @app.post("/ask_trace")
    def ask_trace(payload: dict) -> dict:
        question = payload.get("question", "")
        log.reset()
        t0 = time.perf_counter()
        with app.state.observability.span("ask"):
            rr = app.state.sql_path.run(question)
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "path": rr.path,
            "sql": rr.sql,
            "columns": rr.result.columns if rr.result else None,
            "rows": [list(r) for r in rr.result.rows] if rr.result else None,
            "chart_spec": rr.chart_spec,
            "narrative": rr.narrative,
            "model": model_name,
            "steps": list(log.steps),
            "elapsed_ms": round(elapsed, 1),
        }

    return app


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("ASKLAKE_API_HOST", "0.0.0.0")
    port = int(os.environ.get("ASKLAKE_API_PORT", "8000"))
    uvicorn.run(build_app(), host=host, port=port)
