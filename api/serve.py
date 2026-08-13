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
from contextvars import ContextVar
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from api.deps import require_principal
from api.main import create_app
from engine.auth.static_token import StaticTokenAuthenticator
from engine.governance.audit import AuditLog
from engine.governance.graph import GovernedGraphPath
from engine.governance.policy import GovernanceError, PolicyGovernance, load_policy
from engine.governance.schema import governed_semantic_layer
from engine.governance.views import build_role_views
from engine.graph.intent import IntentResolver
from engine.graph.ontology import load_ontology
from engine.graph.persistence import load_store
from engine.lakehouse.duckdb_backend import DuckDBBackend
from engine.llm.factory import make_provider
from engine.ports.auth import Principal
from engine.ports.llm import LLMProvider
from engine.ports.storage import QueryResult, StorageBackend
from engine.retrieval.agentic_sql_path import AgenticSqlPath
from engine.retrieval.graph_rag_path import GraphRagPath, GroundedGraphRagPath
from engine.retrieval.grounded_sql_path import GroundedSqlPath
from engine.retrieval.router import Router
from engine.retrieval.synthesizer import Synthesizer
from engine.semantic.semantic_layer import SemanticLayerProvider
from engine.semantic.semantic_model import load_semantic_layer
from engine.semantic.value_index import build_value_index

PARQUET_DIR = os.environ.get("ASKLAKE_PARQUET_DIR", "data/imdb/parquet")
SEMANTIC_YAML = "datasets/imdb/semantic.yaml"
GRAPH_PATH = os.environ.get("ASKLAKE_GRAPH_PATH", "data/imdb/graph/triples.jsonl")
GRAPH_BACKEND = os.environ.get("ASKLAKE_GRAPH_BACKEND", "memory").lower()
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
ONTOLOGY_YAML = "datasets/imdb/graph/ontology.yaml"
GOVERNANCE_YAML = "datasets/imdb/governance.yaml"
AUTH_CONFIG = os.environ.get("ASKLAKE_AUTH_CONFIG", "auth.yaml")
AUDIT_PATH = os.environ.get("ASKLAKE_AUDIT_PATH", "data/audit/events.jsonl")


class AskTraceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=4_000)
    path: Literal["auto", "sql", "graph", "fusion"] = "auto"
    provider: str = Field(default="", max_length=100)
    model: str = Field(default="", max_length=200)
    api_key: str = Field(default="", max_length=4_096)


def apply_duckdb_guardrails(
    backend: StorageBackend,
    memory_limit: str = "2GB",
    max_temp_size: str = "4GB",
    threads: int = 2,
) -> None:
    """Cap RAM and spill; a production boot fails if these controls cannot be installed."""
    for stmt in (
        f"SET memory_limit='{memory_limit}'",
        f"SET max_temp_directory_size='{max_temp_size}'",
        f"SET threads={threads}",
    ):
        backend.run_sql(stmt)


def _triples_of(rr) -> list[list] | None:
    """Pull graph triples [subject, relation, object, source] out of a RetrievalResult,
    or None when the result is not a graph triple table (e.g. a SQL result)."""
    if rr and rr.result and rr.result.columns == ["subject", "relation", "object", "source"]:
        return [list(r) for r in rr.result.rows]
    return None


class _TraceLog:
    """Ordered, request-scoped record of backend processing steps (for the UI)."""

    def __init__(self) -> None:
        self._steps: ContextVar[list[dict] | None] = ContextVar("asklake_trace", default=None)

    def _current(self) -> list[dict]:
        current = self._steps.get()
        if current is None:
            current = []
            self._steps.set(current)
        return current

    @property
    def steps(self) -> list[dict]:
        return list(self._current())

    def reset(self) -> None:
        # A fresh mutable list is inherited by LangGraph worker contexts, so steps emitted there
        # remain visible to this request without sharing state with concurrent requests.
        self._steps.set([])

    def add(
        self,
        step: str,
        *,
        ok: bool = True,
        ms: float = 0.0,
        detail: str = "",
        sql: str | None = None,
    ) -> None:
        record = {"step": step, "ok": ok, "ms": round(ms, 1), "detail": detail, "sql": sql}
        self._current().append(record)


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
                detail=f"error: {type(exc).__name__}",
                sql=sql,
            )
            raise

    def list_tables(self):
        return self._inner.list_tables()


class _TracingGraphPath:
    """Wraps a graph path to record a trace step. When wrapping GroundedGraphRagPath the inner
    path makes one LLM call to turn retrieved facts into a natural-language answer."""

    name = "graph"

    def __init__(self, inner, log: _TraceLog) -> None:
        self._inner = inner
        self._log = log

    def can_handle(self, question: str) -> bool:
        return self._inner.can_handle(question)

    def run(self, question: str):
        t0 = time.perf_counter()
        rr = self._inner.run(question)
        n = len(rr.result.rows) if rr.result else 0
        self._log.add(
            "Search knowledge graph",
            ok=True,
            ms=(time.perf_counter() - t0) * 1000,
            detail=f"{n} fact(s) retrieved via multi-hop graph traversal",
        )
        return rr


def build_app(
    llm: LLMProvider | None = None,
    backend: StorageBackend | None = None,
    graph_store=None,
) -> FastAPI:
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

    policy = load_policy(GOVERNANCE_YAML)
    governance = PolicyGovernance(policy, action="raw_sql")

    authenticator = (
        StaticTokenAuthenticator.from_yaml(AUTH_CONFIG, anonymous_role=policy.anonymous_role)
        if AUTH_CONFIG and Path(AUTH_CONFIG).exists()
        else StaticTokenAuthenticator({}, anonymous_role=policy.anonymous_role)
    )
    unknown = authenticator.roles - set(policy.roles)
    if unknown:
        raise ValueError(f"auth.yaml roles not in governance.yaml: {sorted(unknown)}")
    if authenticator.anonymous_role and authenticator.anonymous_role not in policy.roles:
        raise ValueError("anonymous authentication role is not in governance.yaml")
    build_role_views(backend, policy)  # rls_<role> schemas of filtered/redacted views
    path_governance = governance.for_action("ask")

    agent_kind = os.environ.get("ASKLAKE_AGENT", "grounded").lower()
    semantic_layer = load_semantic_layer(SEMANTIC_YAML)
    available_tables = {table.name for table in backend.list_tables()}
    _schemas: dict[str, _TracingSchema] = {}
    _value_indices: dict[str, object | None] = {}

    def _role_backend(role: str) -> StorageBackend:
        return path_governance.scoped_backend(backend, role)

    def _schema_for(role: str) -> _TracingSchema:
        if role not in _schemas:
            role_layer = governed_semantic_layer(
                semantic_layer, policy, role, available_tables=available_tables
            )
            _schemas[role] = _TracingSchema(SemanticLayerProvider(role_layer), log)
        return _schemas[role]

    def _value_index_for(role: str):
        if agent_kind != "grounded":
            return None
        if role not in _value_indices:
            try:
                role_layer = governed_semantic_layer(
                    semantic_layer, policy, role, available_tables=available_tables
                )
                _value_indices[role] = build_value_index(role_layer, _role_backend(role))
            except Exception as exc:  # noqa: BLE001 - optional grounding aid, never authorization
                print(
                    f"[api.serve] value index unavailable for role {role!r} ({exc}); "
                    "continuing without value-linking"
                )
                _value_indices[role] = None
        return _value_indices[role]

    def _make_path(provider_llm: LLMProvider, role: str):
        tllm = _TracingLLM(provider_llm, log)
        rbackend = _TracingBackend(_role_backend(role), log)
        tschema = _schema_for(role)
        if agent_kind == "grounded":
            return GroundedSqlPath(
                tllm,
                tschema,
                rbackend,
                governance=path_governance,
                role=role,
                value_index=_value_index_for(role),
                k_candidates=3,
                max_retries=2,
            )
        return AgenticSqlPath(
            tllm, tschema, rbackend, governance=path_governance, role=role, max_retries=2
        )

    def _model_of(provider_llm: LLMProvider) -> str:
        return getattr(provider_llm, "_model", None) or type(provider_llm).__name__

    _default_paths: dict[str, object] = {}

    def _default_path_for(role: str):
        if default_llm is None:
            return None
        if role not in _default_paths:
            _default_paths[role] = _make_path(default_llm, role)
        return _default_paths[role]

    default_model = _model_of(default_llm) if default_llm is not None else None
    default_provider = type(default_llm).__name__ if default_llm is not None else None

    app = create_app(
        backend=tbackend,
        governance=governance,
        sql_path_resolver=_default_path_for,
        authenticator=authenticator,
        audit=AuditLog(path=AUDIT_PATH or None),
    )
    app.state.trace_log = log
    app.state.model_name = default_model
    app.state.provider_name = default_provider
    app.state.sql_path_kind = agent_kind

    # Knowledge-graph ontology (needed first: relation_roles drive Neo4j node typing).
    graph_attr_relations: frozenset[str] = frozenset()
    graph_connective: frozenset[str] = frozenset()
    graph_empty_hint = "No matching facts found in the knowledge graph."
    intent_resolver = None
    _ontology = None
    try:
        _ontology = load_ontology(ONTOLOGY_YAML)
        graph_attr_relations = frozenset(_ontology.attribute_relations)
        graph_connective = frozenset(_ontology.connective_relations)
        if _ontology.intents:
            intent_resolver = IntentResolver(_ontology)
        if _ontology.empty_graph_hint:
            graph_empty_hint = _ontology.empty_graph_hint
    except Exception as exc:  # noqa: BLE001
        print(f"[api.serve] ontology unavailable ({exc}); seeding without typing")

    # Optional knowledge graph: injected store (tests) > Neo4j (opt-in) > persisted triples.
    neo4j_retriever = None
    if graph_store is None and GRAPH_BACKEND == "neo4j":
        try:
            from neo4j import GraphDatabase

            from engine.graph.neo4j_retriever import Neo4jGraphRetriever
            from engine.graph.neo4j_store import Neo4jGraphStore

            _roles = _ontology.relation_roles if _ontology is not None else {}
            _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            try:
                _driver.verify_connectivity()
                graph_store = Neo4jGraphStore(_driver, relation_roles=_roles)
                neo4j_retriever = Neo4jGraphRetriever(
                    graph_store,
                    intent_resolver,
                    attribute_relations=graph_attr_relations,
                    connective_relations=graph_connective,
                    relation_roles=_roles,
                )
            except Exception:
                _driver.close()
                raise
            print(f"[api.serve] graph backend: neo4j ({NEO4J_URI})")
        except Exception as exc:  # noqa: BLE001
            print(f"[api.serve] neo4j unavailable ({exc}); falling back to in-memory graph")
            graph_store = None
    if graph_store is None and Path(GRAPH_PATH).exists():
        try:
            graph_store = load_store(GRAPH_PATH)
        except Exception as exc:  # noqa: BLE001
            print(f"[api.serve] failed to load graph at {GRAPH_PATH} ({exc}); graph disabled")
            graph_store = None

    _base_graph = None
    if graph_store is not None:
        if neo4j_retriever is not None:
            _base_graph = GraphRagPath(
                graph_store, retriever=neo4j_retriever, empty_hint=graph_empty_hint
            )
        else:
            _base_graph = GraphRagPath(
                graph_store,
                attribute_relations=graph_attr_relations,
                connective_relations=graph_connective,
                intent_resolver=intent_resolver,
                empty_hint=graph_empty_hint,
            )
    _graph_paths: dict[str, object] = {}
    _denied_graph_sources: dict[str, frozenset[str]] = {}

    def _denied_graph_sources_for(role: str) -> frozenset[str]:
        if role not in _denied_graph_sources:
            if policy.row_predicate(role, "title_basics") is None:
                _denied_graph_sources[role] = frozenset()
            else:
                # Graph sources use imdb:<tconst>/wiki:<tconst>. Taking the difference between the
                # base catalog and the role view carries SQL row policy into a pre-existing graph,
                # so an older graph build cannot re-expose titles now filtered by content policy.
                denied = backend.run_sql(
                    f'SELECT tconst FROM main."title_basics" EXCEPT '
                    f'SELECT tconst FROM "rls_{role}"."title_basics"'
                )
                _denied_graph_sources[role] = frozenset(
                    source
                    for (tconst,) in denied.rows
                    for source in (f"imdb:{tconst}", f"wiki:{tconst}")
                )
        return _denied_graph_sources[role]

    def _graph_path_for(role: str):
        if _base_graph is None:
            return None
        if role not in _graph_paths:
            governed_graph = GovernedGraphPath(
                _base_graph,
                policy,
                role,
                denied_sources=_denied_graph_sources_for(role),
            )
            graph_impl = governed_graph
            if default_llm is not None:
                # Governance is deliberately inside the LLM wrapper: denied relations and missing
                # citations never enter the model prompt.
                graph_impl = GroundedGraphRagPath(governed_graph, default_llm)
            _graph_paths[role] = _TracingGraphPath(graph_impl, log)
        return _graph_paths[role]

    routing_graph_path = _graph_path_for(policy.anonymous_role)
    synth = Synthesizer()
    # router.route() only scores the question — it never calls sql_path.run — so passing
    # default_path (which may be None when there is no boot key) is safe. Execution below uses
    # the per-request sql_runner from _resolve(), so a BYO key is honoured, not default_path.
    router = Router(
        _default_path_for(policy.anonymous_role),
        routing_graph_path,
        synth,
        entity_vocab=graph_store.entities() if graph_store is not None else (),
    )
    app.state.graph_enabled = _base_graph is not None

    def _decide(requested: str, question: str):
        """(paths, fuse): honor an explicit override, else the heuristic Router."""
        if requested == "sql":
            return ("sql",), False
        if requested == "graph":
            return ("graph",), False
        if requested == "fusion":
            return ("sql", "graph"), True
        d = router.route(question)
        return d.paths, d.fuse

    def _resolve(provider: str, model: str, api_key: str, role: str):
        """Return (path, effective_model) for this request, or (None, None) when no key is usable.

        A typed key => build a per-request provider (BYO). Otherwise fall back to the boot-time
        default provider; when there is none, return (None, None) so the caller can prompt the
        user to enter a key. (The UI always sends `provider` from the selectbox, so keying off
        `api_key` is what distinguishes "bring your own" from "use the server default".)"""
        if api_key:
            req_llm = make_provider(provider or None, api_key=api_key or None, model=model or None)
            return _make_path(req_llm, role), _model_of(req_llm)
        dp = _default_path_for(role)
        if dp is not None:
            return dp, default_model
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

    @app.get("/info")
    def info() -> dict:
        return {
            "provider": default_provider or "(client-supplied)",
            "model": default_model or "(set in the sidebar)",
            "path": f"semantic-grounded + self-correcting ({agent_kind})",
        }

    @app.post("/ask_trace")
    def ask_trace(
        payload: AskTraceRequest,
        request: Request,
        principal: Principal = Depends(require_principal),  # noqa: B008
    ):
        role = principal.role
        question = payload.question
        api_key = payload.api_key.strip()
        requested = payload.path
        log.reset()

        def early_failure(
            reason_code: str,
            narrative: str,
            *,
            action: str = "ask",
            model_name: str = "",
            result_path: str = "sql",
        ) -> dict:
            app.state.observability.event("ask_error", reason_code=reason_code)
            app.state.audit.write(
                event="query",
                user=principal.user,
                role=role,
                action=action,
                path="/ask_trace",
                decision="failed",
                reason_code=reason_code,
                request_id=request.state.request_id,
                query_text=question,
            )
            return {
                **_empty(narrative, model_name),
                "path": result_path,
                "request_id": request.state.request_id,
            }

        paths, fuse = _decide(requested, question)
        needs_graph = "graph" in paths
        request_graph_path = _graph_path_for(role) if needs_graph else None
        if needs_graph and request_graph_path is None:  # graph asked for but not built
            if "sql" in paths:
                paths, fuse, needs_graph = ("sql",), False, False
            else:
                return early_failure(
                    "graph_unavailable",
                    "The knowledge graph isn't built yet — run `make build-graph`, "
                    "or ask a SQL-style question.",
                    action="graph",
                    result_path="graph",
                )

        needs_sql = "sql" in paths
        sql_runner = None
        model = ""
        if needs_sql:
            try:
                sql_runner, model = _resolve(payload.provider, payload.model, api_key, role)
            except Exception:  # noqa: BLE001 - client gets a stable error; audit has request ID
                return early_failure(
                    "model_initialization_failed",
                    "Could not initialize the model. Verify the provider, model and API key. "
                    f"Request ID: {request.state.request_id}",
                    model_name=payload.model,
                )
            if sql_runner is None:
                return early_failure(
                    "model_unavailable", "Enter your API key in the sidebar to ask questions."
                )

        t0 = time.perf_counter()
        graph_triples = None
        try:
            with app.state.observability.span("ask", role=role):
                if fuse:
                    sql_rr = sql_runner.run(question)
                    graph_rr = request_graph_path.run(question)
                    rr = synth.fuse(question, [sql_rr, graph_rr])
                    graph_triples = _triples_of(graph_rr)
                elif needs_graph:
                    rr = request_graph_path.run(question)
                    graph_triples = _triples_of(rr)
                else:
                    rr = sql_runner.run(question)
        except GovernanceError as exc:
            app.state.observability.event(f"access.{role}.blocked")
            app.state.audit.write(
                event="authorization",
                user=principal.user,
                role=role,
                action="graph" if needs_graph and not needs_sql else "ask",
                path="/ask_trace",
                decision="denied",
                reason_code=exc.code,
                request_id=request.state.request_id,
                query_text=question,
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
            log.add("Request failed", ok=False, detail=type(exc).__name__)
            app.state.observability.event("ask_error", error_type=type(exc).__name__)
            app.state.audit.write(
                event="query",
                user=principal.user,
                role=role,
                action="ask",
                path="/ask_trace",
                decision="failed",
                reason_code="request_failed",
                request_id=request.state.request_id,
                query_text=question,
            )
            return {
                **_empty(
                    "The request failed. Check your API key and model, then retry. "
                    f"Request ID: {request.state.request_id}",
                    model,
                ),
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
                "request_id": request.state.request_id,
            }
        elapsed = (time.perf_counter() - t0) * 1000
        decision = "allowed" if rr.result is not None else "failed"
        app.state.observability.event(f"access.{role}.{decision}")
        app.state.audit.write(
            event="query",
            user=principal.user,
            role=role,
            action="graph" if needs_graph and not needs_sql else "ask",
            path=rr.path,
            decision=decision,
            row_count=len(rr.result.rows) if rr.result else 0,
            request_id=request.state.request_id,
            query_text=question,
        )
        governance_metadata = path_governance.response_metadata(role, rr.sql)
        if needs_graph:
            graph_licenses = sorted(
                {
                    item.classification.license
                    for relation, item in policy.graph_relations.items()
                    if relation in policy.graph_relations_for(role) and item.classification.license
                }
            )
            governance_metadata["licenses"] = sorted(
                set(governance_metadata["licenses"]) | set(graph_licenses)
            )
            governance_metadata["notices"] = [
                policy.license_notices[name]
                for name in governance_metadata["licenses"]
                if name in policy.license_notices
            ]
        narrative = rr.narrative
        if needs_sql and rr.result is None:
            narrative = f"The query could not be completed. Request ID: {request.state.request_id}"
        return {
            "path": rr.path,
            "sql": rr.sql,
            "columns": rr.result.columns if rr.result else None,
            "rows": [list(r) for r in rr.result.rows] if rr.result else None,
            "chart_spec": rr.chart_spec,
            "narrative": narrative,
            "graph_triples": graph_triples,
            "model": model,
            "steps": list(log.steps),
            "elapsed_ms": round(elapsed, 1),
            "governance": governance_metadata,
            "request_id": request.state.request_id,
        }

    return app


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("ASKLAKE_API_HOST", "0.0.0.0")
    port = int(os.environ.get("ASKLAKE_API_PORT", "8000"))
    uvicorn.run(build_app(), host=host, port=port)
