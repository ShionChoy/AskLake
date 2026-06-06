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
    """Build the production app. Always serves /health, /query, /info and /ask_trace; the LLM is
    optional at boot — the browser can supply a provider/key per request (BYO key)."""
    if backend is None:
        parquet = PARQUET_DIR if Path(PARQUET_DIR).exists() else None
        backend = DuckDBBackend(parquet_dir=parquet)
        apply_duckdb_guardrails(backend)

    # Boot-time default provider: explicit arg, else env, else none (BYO key via the UI).
    default_llm = llm
    if default_llm is None:
        try:
            default_llm = make_provider()
        except Exception as exc:  # noqa: BLE001
            print(f"[api.serve] no default LLM provider ({exc}); supply a key in the UI sidebar")
            default_llm = None

    log = _TraceLog()
    tbackend = _TracingBackend(backend, log)
    tschema = _TracingSchema(SemanticLayerProvider.from_yaml(SEMANTIC_YAML), log)

    def _make_path(provider_llm: LLMProvider) -> AgenticSqlPath:
        return AgenticSqlPath(_TracingLLM(provider_llm, log), tschema, tbackend, max_retries=2)

    def _model_of(provider_llm: LLMProvider) -> str:
        return getattr(provider_llm, "_model", None) or type(provider_llm).__name__

    default_path = _make_path(default_llm) if default_llm is not None else None
    default_model = _model_of(default_llm) if default_llm is not None else None
    default_provider = type(default_llm).__name__ if default_llm is not None else None

    app = create_app(backend=tbackend, sql_path=default_path)
    app.state.trace_log = log
    app.state.model_name = default_model
    app.state.provider_name = default_provider

    def _resolve(provider: str, model: str, api_key: str):
        """Return (path, effective_model) for this request, or (None, None) when no key is usable.

        A typed key => build a per-request provider (BYO). Otherwise fall back to the boot-time
        default provider; when there is none, return (None, None) so the caller can prompt the
        user to enter a key. (The UI always sends `provider` from the selectbox, so keying off
        `api_key` is what distinguishes "bring your own" from "use the server default".)"""
        if api_key:
            req_llm = make_provider(provider or None, api_key=api_key or None, model=model or None)
            return _make_path(req_llm), _model_of(req_llm)
        if default_path is not None:
            return default_path, default_model
        return None, None

    def _empty(narrative: str, model: str = "") -> dict:
        return {
            "path": "sql",
            "sql": "",
            "columns": None,
            "rows": None,
            "chart_spec": None,
            "narrative": narrative,
            "model": model,
            "steps": list(log.steps),
            "elapsed_ms": 0.0,
        }

    def _redact(text: str, secret: str) -> str:
        """Keep the (useful) error detail but never echo the user's key back to the client."""
        return text.replace(secret, "***") if secret else text

    @app.get("/info")
    def info() -> dict:
        return {
            "provider": default_provider or "(client-supplied)",
            "model": default_model or "(set in the sidebar)",
            "path": "semantic-grounded + self-correcting (agentic)",
        }

    @app.post("/ask_trace")
    def ask_trace(payload: dict) -> dict:
        question = payload.get("question", "")
        api_key = payload.get("api_key", "")
        # Single-user, local-first: the shared trace log is reset per request.
        log.reset()
        try:
            path, model = _resolve(payload.get("provider", ""), payload.get("model", ""), api_key)
        except Exception as exc:  # noqa: BLE001
            return _empty(
                _redact(f"Could not initialize the model: {exc}", api_key),
                payload.get("model", ""),
            )
        if path is None:
            return _empty("Enter your API key in the sidebar to ask questions.")
        t0 = time.perf_counter()
        try:
            with app.state.observability.span("ask"):
                rr = path.run(question)
        except Exception as exc:  # noqa: BLE001
            log.add("Model call failed", ok=False, detail=_redact(str(exc), api_key))
            return {
                **_empty(
                    _redact(
                        f"The model call failed: {exc}. Check your API key and model.",
                        api_key,
                    ),
                    model,
                ),
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
            }
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "path": rr.path,
            "sql": rr.sql,
            "columns": rr.result.columns if rr.result else None,
            "rows": [list(r) for r in rr.result.rows] if rr.result else None,
            "chart_spec": rr.chart_spec,
            "narrative": rr.narrative,
            "model": model,
            "steps": list(log.steps),
            "elapsed_ms": round(elapsed, 1),
        }

    return app


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("ASKLAKE_API_HOST", "0.0.0.0")
    port = int(os.environ.get("ASKLAKE_API_PORT", "8000"))
    uvicorn.run(build_app(), host=host, port=port)
