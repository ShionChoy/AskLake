from __future__ import annotations

import uuid
from collections.abc import Callable

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from api.deps import require_principal
from engine.auth.static_token import StaticTokenAuthenticator
from engine.governance.audit import AuditLog
from engine.governance.passthrough import PassthroughGovernance
from engine.governance.policy import GovernanceError
from engine.lakehouse.duckdb_backend import DuckDBBackend
from engine.observability.noop import NoopObservability
from engine.ports.auth import Authenticator, Principal
from engine.ports.governance import GovernanceHook
from engine.ports.observability import Observability
from engine.ports.retrieval import RetrievalPath
from engine.ports.storage import StorageBackend
from engine.settings import get_settings


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sql: str = Field(min_length=1, max_length=20_000)


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=4_000)


def create_app(
    backend: StorageBackend | None = None,
    governance: GovernanceHook | None = None,
    observability: Observability | None = None,
    sql_path: RetrievalPath | None = None,
    sql_path_resolver: Callable[[str], RetrievalPath | None] | None = None,
    authenticator: Authenticator | None = None,
    audit: AuditLog | None = None,
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
    app.state.sql_path_resolver = sql_path_resolver
    app.state.authenticator = authenticator or StaticTokenAuthenticator({})
    app.state.audit = audit or AuditLog()

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": "asklake"}

    @app.post("/query")
    def query(
        req: QueryRequest,
        request: Request,
        principal: Principal = Depends(require_principal),  # noqa: B008
    ):
        role = principal.role
        with observability.span("query", role=role):
            try:
                sql = governance.before_query(req.sql, role=role)
                scope = getattr(governance, "scoped_backend", None)
                governed_backend = scope(backend, role) if scope is not None else backend
                result = governed_backend.run_sql(sql)
                result = governance.after_result(result, role=role)
            except GovernanceError as exc:
                observability.event("query_denied", code=exc.code)
                observability.event(f"access.{role}.blocked")
                app.state.audit.write(
                    event="authorization",
                    user=principal.user,
                    role=role,
                    action="raw_sql",
                    path="/query",
                    decision="denied",
                    reason_code=exc.code,
                    request_id=request.state.request_id,
                    query_text=req.sql,
                )
                return JSONResponse(
                    status_code=exc.status_code,
                    content={
                        "error": str(exc),
                        "code": exc.code,
                        "request_id": request.state.request_id,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                observability.event("query_error", error_type=type(exc).__name__)
                observability.event(f"access.{role}.blocked")
                app.state.audit.write(
                    event="query",
                    user=principal.user,
                    role=role,
                    action="raw_sql",
                    path="/query",
                    decision="failed",
                    reason_code="query_execution_failed",
                    request_id=request.state.request_id,
                    query_text=req.sql,
                )
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "query execution failed",
                        "code": "query_execution_failed",
                        "request_id": request.state.request_id,
                    },
                )
        observability.event(f"access.{role}.allowed")
        app.state.audit.write(
            event="query",
            user=principal.user,
            role=role,
            action="raw_sql",
            path="/query",
            decision="allowed",
            row_count=len(result.rows),
            request_id=request.state.request_id,
            query_text=req.sql,
        )
        metadata = getattr(governance, "response_metadata", None)
        return {
            "columns": result.columns,
            "rows": [list(r) for r in result.rows],
            "governance": metadata(role, req.sql) if metadata else None,
            "request_id": request.state.request_id,
        }

    @app.post("/ask")
    def ask(
        req: AskRequest,
        request: Request,
        principal: Principal = Depends(require_principal),  # noqa: B008
    ):
        role = principal.role
        resolver = app.state.sql_path_resolver
        sp: RetrievalPath | None = resolver(role) if resolver is not None else app.state.sql_path
        ask_governance = (
            governance.for_action("ask") if hasattr(governance, "for_action") else governance
        )
        if sp is None and resolver is not None:
            app.state.audit.write(
                event="query",
                user=principal.user,
                role=role,
                action="ask",
                path="/ask",
                decision="failed",
                reason_code="model_unavailable",
                request_id=request.state.request_id,
                query_text=req.question,
            )
            return JSONResponse(
                status_code=503,
                content={
                    "error": "no server-side language model is configured",
                    "code": "model_unavailable",
                    "request_id": request.state.request_id,
                },
            )
        if sp is None:
            from engine.llm.anthropic_provider import AnthropicProvider
            from engine.retrieval.sql_path import SqlPath
            from engine.semantic.raw_schema import RawSchemaProvider

            llm = AnthropicProvider(model=settings.llm_model)
            # Build per-request: role is per-caller, so caching would pin the first caller's role.
            scope = getattr(ask_governance, "scoped_backend", None)
            ask_backend = scope(backend, role) if scope is not None else backend
            sp = SqlPath(
                llm,
                RawSchemaProvider(ask_backend),
                ask_backend,
                ask_governance,
                role=role,
            )
        with observability.span("ask", role=role):
            try:
                rr = sp.run(req.question)
            except GovernanceError as exc:
                observability.event(f"access.{role}.blocked")
                app.state.audit.write(
                    event="authorization",
                    user=principal.user,
                    role=role,
                    action="ask",
                    path="/ask",
                    decision="denied",
                    reason_code=exc.code,
                    request_id=request.state.request_id,
                    query_text=req.question,
                )
                return JSONResponse(
                    status_code=exc.status_code,
                    content={
                        "error": str(exc),
                        "code": exc.code,
                        "request_id": request.state.request_id,
                    },
                )
        decision = "allowed" if rr.result is not None else "failed"
        observability.event(f"access.{role}.{decision}")
        app.state.audit.write(
            event="query",
            user=principal.user,
            role=role,
            action="ask",
            path=rr.path,
            decision=decision,
            row_count=len(rr.result.rows) if rr.result else 0,
            request_id=request.state.request_id,
            query_text=req.question,
        )
        metadata = getattr(ask_governance, "response_metadata", None)
        narrative = rr.narrative
        if rr.result is None and rr.path == "sql":
            narrative = f"The query could not be completed. Request ID: {request.state.request_id}"
        return {
            "path": rr.path,
            "sql": rr.sql,
            "columns": rr.result.columns if rr.result else None,
            "rows": [list(r) for r in rr.result.rows] if rr.result else None,
            "chart_spec": rr.chart_spec,
            "narrative": narrative,
            "governance": metadata(role, rr.sql) if metadata else None,
            "request_id": request.state.request_id,
        }

    if hasattr(observability, "registry"):
        from fastapi.responses import Response
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        @app.get("/metrics")
        def metrics() -> Response:
            return Response(generate_latest(observability.registry), media_type=CONTENT_TYPE_LATEST)

    return app
