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

from fastapi import Depends, FastAPI

from api.deps import require_principal
from api.main import create_app
from engine.auth.static_token import StaticTokenAuthenticator
from engine.governance.policy import PolicyGovernance, load_policy
from engine.governance.views import build_role_views
from engine.graph.intent import IntentResolver
from engine.graph.ontology import load_ontology
from engine.graph.persistence import load_store
from engine.lakehouse.duckdb_backend import DuckDBBackend
from engine.lakehouse.role_scoped_backend import RoleScopedBackend
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
SEMANTIC_YAML = "datasets/imdb_cmu/semantic.yaml"
GRAPH_PATH = os.environ.get("ASKLAKE_GRAPH_PATH", "data/imdb/graph/triples.jsonl")
ONTOLOGY_YAML = "datasets/imdb_cmu/graph/ontology.yaml"
GOVERNANCE_YAML = "datasets/imdb_cmu/governance.yaml"
AUTH_CONFIG = os.environ.get("ASKLAKE_AUTH_CONFIG", "auth.yaml")


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


def _triples_of(rr) -> list[list] | None:
    """Pull graph triples [subject, relation, object, source] out of a RetrievalResult,
    or None when the result is not a graph triple table (e.g. a SQL result)."""
    if rr and rr.result and rr.result.columns == ["subject", "relation", "object", "source"]:
        return [list(r) for r in rr.result.rows]
    return None


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
    tschema = _TracingSchema(SemanticLayerProvider.from_yaml(SEMANTIC_YAML), log)

    agent_kind = os.environ.get("ASKLAKE_AGENT", "grounded").lower()
    value_index = None
    if agent_kind == "grounded":
        try:
            value_index = build_value_index(load_semantic_layer(SEMANTIC_YAML), backend)
        except Exception as exc:  # noqa: BLE001 - degrade to no value hints
            print(f"[api.serve] value index unavailable ({exc}); continuing without value-linking")

    policy = load_policy(GOVERNANCE_YAML)
    governance = PolicyGovernance(policy)
    build_role_views(backend, policy)  # rls_<role> schemas of filtered/redacted views

    # Paths generate SQL via LLM; forbid writes but don't require LIMIT (the memory guardrail
    # and view-level RLS already protect the engine). LIMIT enforcement stays on /query (raw SQL).
    from engine.governance.policy import Policy as _Policy  # noqa: PLC0415

    path_governance = PolicyGovernance(
        _Policy(
            pii_columns=policy.pii_columns,
            mask_roles=policy.mask_roles,
            row_filters=policy.row_filters,
            roles=policy.roles,
            row_security=policy.row_security,
            require_limit=False,
            forbid_writes=policy.forbid_writes,
        )
    )

    authenticator = (
        StaticTokenAuthenticator.from_yaml(AUTH_CONFIG)
        if AUTH_CONFIG and Path(AUTH_CONFIG).exists()
        else StaticTokenAuthenticator({})
    )
    unknown = authenticator.roles - set(policy.roles)
    if unknown:
        raise ValueError(f"auth.yaml roles not in governance.yaml: {sorted(unknown)}")

    def _make_path(provider_llm: LLMProvider, role: str):
        tllm = _TracingLLM(provider_llm, log)
        rbackend = _TracingBackend(RoleScopedBackend(backend, role), log)
        if agent_kind == "grounded":
            return GroundedSqlPath(
                tllm,
                tschema,
                rbackend,
                governance=path_governance,
                role=role,
                value_index=value_index,
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
        sql_path=_default_path_for("public"),
        authenticator=authenticator,
    )
    app.state.trace_log = log
    app.state.model_name = default_model
    app.state.provider_name = default_provider
    app.state.sql_path_kind = agent_kind

    # Optional knowledge graph: use an injected store (tests) or load the persisted triples.
    if graph_store is None and Path(GRAPH_PATH).exists():
        try:
            graph_store = load_store(GRAPH_PATH)
        except Exception as exc:  # noqa: BLE001
            print(f"[api.serve] failed to load graph at {GRAPH_PATH} ({exc}); graph disabled")
            graph_store = None
    graph_attr_relations: frozenset[str] = frozenset()
    graph_connective: frozenset[str] = frozenset()
    graph_empty_hint = "No matching facts found in the knowledge graph."
    intent_resolver = None
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

    if graph_store is not None:
        _base_graph = GraphRagPath(
            graph_store,
            attribute_relations=graph_attr_relations,
            connective_relations=graph_connective,
            intent_resolver=intent_resolver,
            empty_hint=graph_empty_hint,
        )
        _graph_impl = _base_graph
        if default_llm is not None:
            _graph_impl = GroundedGraphRagPath(_base_graph, default_llm)
        graph_path = _TracingGraphPath(_graph_impl, log)
    else:
        graph_path = None
    synth = Synthesizer()
    # router.route() only scores the question — it never calls sql_path.run — so passing
    # default_path (which may be None when there is no boot key) is safe. Execution below uses
    # the per-request sql_runner from _resolve(), so a BYO key is honoured, not default_path.
    router = Router(
        _default_path_for("public"),
        graph_path,
        synth,
        entity_vocab=graph_store.entities() if graph_store is not None else (),
    )
    app.state.graph_enabled = graph_path is not None

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

    def _redact(text: str, secret: str) -> str:
        """Keep the (useful) error detail but never echo the user's key back to the client."""
        return text.replace(secret, "***") if secret else text

    @app.get("/info")
    def info() -> dict:
        return {
            "provider": default_provider or "(client-supplied)",
            "model": default_model or "(set in the sidebar)",
            "path": f"semantic-grounded + self-correcting ({agent_kind})",
        }

    @app.post("/ask_trace")
    def ask_trace(payload: dict, principal: Principal = Depends(require_principal)) -> dict:  # noqa: B008
        role = principal.role
        question = payload.get("question", "")
        api_key = (payload.get("api_key", "") or "").strip()
        requested = (payload.get("path") or "auto").lower()
        # Single-user, local-first: the shared trace log is reset per request.
        log.reset()

        paths, fuse = _decide(requested, question)
        needs_graph = "graph" in paths
        if needs_graph and graph_path is None:  # graph asked for but not built
            if "sql" in paths:
                paths, fuse, needs_graph = ("sql",), False, False
            else:
                return {
                    **_empty(
                        "The knowledge graph isn't built yet — run `make build-graph`, "
                        "or ask a SQL-style question."
                    ),
                    "path": "graph",
                }

        needs_sql = "sql" in paths
        sql_runner = None
        model = ""
        if needs_sql:
            try:
                sql_runner, model = _resolve(
                    payload.get("provider", ""), payload.get("model", ""), api_key, role
                )
            except Exception as exc:  # noqa: BLE001
                return _empty(
                    _redact(f"Could not initialize the model: {exc}", api_key),
                    payload.get("model", ""),
                )
            if sql_runner is None:
                return _empty("Enter your API key in the sidebar to ask questions.")

        t0 = time.perf_counter()
        graph_triples = None
        try:
            with app.state.observability.span("ask"):
                if fuse:
                    sql_rr = sql_runner.run(question)
                    graph_rr = graph_path.run(question)
                    rr = synth.fuse(question, [sql_rr, graph_rr])
                    graph_triples = _triples_of(graph_rr)
                elif needs_graph:
                    rr = graph_path.run(question)
                    graph_triples = _triples_of(rr)
                else:
                    rr = sql_runner.run(question)
        except Exception as exc:  # noqa: BLE001
            log.add("Model call failed", ok=False, detail=_redact(str(exc), api_key))
            return {
                **_empty(
                    _redact(
                        f"The model call failed: {exc}. Check your API key and model.", api_key
                    ),
                    model,
                ),
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
            }
        elapsed = (time.perf_counter() - t0) * 1000
        app.state.observability.event(f"access.{role}.allowed")
        app.state.audit.write(
            user=principal.user,
            role=role,
            path=rr.path,
            decision="allowed",
            row_count=len(rr.result.rows) if rr.result else 0,
            question=question,
        )
        return {
            "path": rr.path,
            "sql": rr.sql,
            "columns": rr.result.columns if rr.result else None,
            "rows": [list(r) for r in rr.result.rows] if rr.result else None,
            "chart_spec": rr.chart_spec,
            "narrative": rr.narrative,
            "graph_triples": graph_triples,
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
