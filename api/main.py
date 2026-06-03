# api/main.py
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from engine.governance.passthrough import PassthroughGovernance
from engine.lakehouse.duckdb_backend import DuckDBBackend
from engine.observability.noop import NoopObservability
from engine.ports.governance import GovernanceHook
from engine.ports.observability import Observability
from engine.ports.storage import StorageBackend
from engine.settings import get_settings


class QueryRequest(BaseModel):
    sql: str
    role: str = "analyst"


def create_app(
    backend: StorageBackend | None = None,
    governance: GovernanceHook | None = None,
    observability: Observability | None = None,
) -> FastAPI:
    if backend is None:
        settings = get_settings()
        backend = DuckDBBackend(parquet_dir=settings.parquet_dir)
    governance = governance or PassthroughGovernance()
    observability = observability or NoopObservability()

    app = FastAPI(title="AskLake", version="0.0.0")
    app.state.backend = backend
    app.state.governance = governance
    app.state.observability = observability

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": "asklake"}

    @app.post("/query")
    def query(req: QueryRequest):
        obs: Observability = app.state.observability
        gov: GovernanceHook = app.state.governance
        with obs.span("query", role=req.role):
            try:
                sql = gov.before_query(req.sql, role=req.role)
                result = backend.run_sql(sql)
                result = gov.after_result(result, role=req.role)
            except Exception as exc:  # noqa: BLE001 - surface SQL/exec errors to client
                obs.event("query_error", error=str(exc))
                return JSONResponse(status_code=400, content={"error": str(exc)})
        return {"columns": result.columns, "rows": [list(r) for r in result.rows]}

    return app
