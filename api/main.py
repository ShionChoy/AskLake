from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from engine.governance.passthrough import PassthroughGovernance
from engine.lakehouse.duckdb_backend import DuckDBBackend
from engine.observability.noop import NoopObservability
from engine.ports.governance import GovernanceHook
from engine.ports.observability import Observability
from engine.ports.retrieval import RetrievalPath
from engine.ports.storage import StorageBackend
from engine.settings import get_settings


class QueryRequest(BaseModel):
    sql: str
    role: str = "analyst"


class AskRequest(BaseModel):
    question: str
    role: str = "analyst"


def create_app(
    backend: StorageBackend | None = None,
    governance: GovernanceHook | None = None,
    observability: Observability | None = None,
    sql_path: RetrievalPath | None = None,
) -> FastAPI:
    settings = get_settings()
    if backend is None:
        backend = DuckDBBackend(parquet_dir=settings.parquet_dir)
    governance = governance or PassthroughGovernance()
    observability = observability or NoopObservability()

    app = FastAPI(title="AskLake", version="0.0.0")
    app.state.backend = backend
    app.state.governance = governance
    app.state.observability = observability
    app.state.sql_path = sql_path

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": "asklake"}

    @app.post("/query")
    def query(req: QueryRequest):
        with observability.span("query", role=req.role):
            try:
                sql = governance.before_query(req.sql, role=req.role)
                result = backend.run_sql(sql)
                result = governance.after_result(result, role=req.role)
            except Exception as exc:  # noqa: BLE001
                observability.event("query_error", error=str(exc))
                return JSONResponse(status_code=400, content={"error": str(exc)})
        return {"columns": result.columns, "rows": [list(r) for r in result.rows]}

    @app.post("/ask")
    def ask(req: AskRequest):
        sp: RetrievalPath | None = app.state.sql_path
        if sp is None:
            # Lazy default wiring so importing the app needs no API key.
            from engine.llm.anthropic_provider import AnthropicProvider
            from engine.retrieval.sql_path import SqlPath
            from engine.semantic.raw_schema import RawSchemaProvider

            llm = AnthropicProvider(model=settings.llm_model)
            sp = SqlPath(llm, RawSchemaProvider(backend), backend, governance, role=req.role)
            app.state.sql_path = sp
        with observability.span("ask", role=req.role):
            rr = sp.run(req.question)
        return {
            "path": rr.path,
            "sql": rr.sql,
            "columns": rr.result.columns if rr.result else None,
            "rows": [list(r) for r in rr.result.rows] if rr.result else None,
            "chart_spec": rr.chart_spec,
            "narrative": rr.narrative,
        }

    return app
