from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from api.deps import require_principal
from engine.auth.static_token import StaticTokenAuthenticator
from engine.governance.audit import AuditLog
from engine.governance.passthrough import PassthroughGovernance
from engine.lakehouse.duckdb_backend import DuckDBBackend
from engine.observability.noop import NoopObservability
from engine.ports.auth import Authenticator, Principal
from engine.ports.governance import GovernanceHook
from engine.ports.observability import Observability
from engine.ports.retrieval import RetrievalPath
from engine.ports.storage import StorageBackend
from engine.settings import get_settings


class QueryRequest(BaseModel):
    sql: str


class AskRequest(BaseModel):
    question: str


def create_app(
    backend: StorageBackend | None = None,
    governance: GovernanceHook | None = None,
    observability: Observability | None = None,
    sql_path: RetrievalPath | None = None,
    authenticator: Authenticator | None = None,
) -> FastAPI:
    settings = get_settings()
    if backend is None:
        backend = DuckDBBackend(parquet_dir=settings.parquet_dir)
    governance = governance or PassthroughGovernance()
    if observability is None:
        if settings.observability_backend == "prometheus":
            from engine.observability.prometheus import PrometheusObservability

            observability = PrometheusObservability()
        else:
            observability = NoopObservability()

    app = FastAPI(title="AskLake", version="0.0.0")
    app.state.backend = backend
    app.state.governance = governance
    app.state.observability = observability
    app.state.sql_path = sql_path
    app.state.authenticator = authenticator or StaticTokenAuthenticator({})
    app.state.audit = AuditLog()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": "asklake"}

    @app.post("/query")
    def query(req: QueryRequest, principal: Principal = Depends(require_principal)):  # noqa: B008
        role = principal.role
        with observability.span("query", role=role):
            try:
                sql = governance.before_query(req.sql, role=role)
                result = backend.run_sql(sql)
                result = governance.after_result(result, role=role)
            except Exception as exc:  # noqa: BLE001
                observability.event("query_error", error=str(exc))
                app.state.observability.event(f"access.{role}.blocked")
                app.state.audit.write(
                    user=principal.user,
                    role=role,
                    path="query",
                    decision="blocked",
                    reason=str(exc),
                    question=req.sql,
                )
                return JSONResponse(status_code=400, content={"error": str(exc)})
        app.state.observability.event(f"access.{role}.allowed")
        app.state.audit.write(
            user=principal.user,
            role=role,
            path="query",
            decision="allowed",
            row_count=len(result.rows),
            question=req.sql,
        )
        return {"columns": result.columns, "rows": [list(r) for r in result.rows]}

    @app.post("/ask")
    def ask(req: AskRequest, principal: Principal = Depends(require_principal)):  # noqa: B008
        role = principal.role
        sp: RetrievalPath | None = app.state.sql_path
        if sp is None:
            from engine.llm.anthropic_provider import AnthropicProvider
            from engine.retrieval.sql_path import SqlPath
            from engine.semantic.raw_schema import RawSchemaProvider

            llm = AnthropicProvider(model=settings.llm_model)
            sp = SqlPath(llm, RawSchemaProvider(backend), backend, governance, role=role)
            app.state.sql_path = sp
        with observability.span("ask", role=role):
            rr = sp.run(req.question)
        app.state.observability.event(f"access.{role}.allowed")
        app.state.audit.write(
            user=principal.user,
            role=role,
            path=rr.path,
            decision="allowed",
            row_count=len(rr.result.rows) if rr.result else 0,
            question=req.question,
        )
        return {
            "path": rr.path,
            "sql": rr.sql,
            "columns": rr.result.columns if rr.result else None,
            "rows": [list(r) for r in rr.result.rows] if rr.result else None,
            "chart_spec": rr.chart_spec,
            "narrative": rr.narrative,
        }

    if hasattr(observability, "registry"):
        from fastapi.responses import Response
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        @app.get("/metrics")
        def metrics() -> Response:
            return Response(generate_latest(observability.registry), media_type=CONTENT_TYPE_LATEST)

    return app
