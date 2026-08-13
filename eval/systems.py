from __future__ import annotations

from engine.agents.agentic_graph import build_agentic_sql_graph
from engine.agents.graph import build_sql_graph
from engine.agents.grounded_graph import build_grounded_sql_graph
from engine.ports.llm import LLMProvider
from engine.ports.storage import QueryResult, StorageBackend
from engine.semantic.raw_schema import RawSchemaProvider
from engine.semantic.semantic_layer import SemanticLayerProvider
from engine.semantic.semantic_model import SemanticLayer


def run_baseline(llm: LLMProvider, backend: StorageBackend, question: str) -> tuple[str, int]:
    """Naive single-prompt baseline: one shot with no self-correction."""
    graph = build_sql_graph(llm, RawSchemaProvider(backend))
    state = graph.invoke({"question": question})
    return state.get("sql", ""), 0


def run_agentic(
    llm: LLMProvider, backend: StorageBackend, question: str, max_retries: int = 2
) -> tuple[str, int]:
    """Return the self-correcting system's final SQL and number of corrections."""

    def executor(sql: str) -> QueryResult:
        return backend.run_sql(sql)

    graph = build_agentic_sql_graph(llm, RawSchemaProvider(backend), executor, max_retries)
    state = graph.invoke({"question": question})
    return state.get("sql", ""), state.get("attempts", 0)


def run_semantic(
    llm: LLMProvider,
    backend: StorageBackend,
    question: str,
    layer: SemanticLayer,
    max_retries: int = 2,
) -> tuple[str, int]:
    """Self-correcting system grounded by the semantic layer instead of raw schema."""

    def executor(sql: str) -> QueryResult:
        return backend.run_sql(sql)

    graph = build_agentic_sql_graph(llm, SemanticLayerProvider(layer), executor, max_retries)
    state = graph.invoke({"question": question})
    return state.get("sql", ""), state.get("attempts", 0)


def _grounded(llm, backend, question, layer, value_index, *, use_plan, use_critic, k, max_retries):
    def executor(sql: str) -> QueryResult:
        return backend.run_sql(sql)

    graph = build_grounded_sql_graph(
        llm,
        SemanticLayerProvider(layer),
        executor,
        value_index=value_index,
        k_candidates=k,
        max_retries=max_retries,
        use_plan=use_plan,
        use_critic=use_critic,
    )
    state = graph.invoke({"question": question})
    return state.get("sql", ""), state.get("attempts", 0)


def run_value_link(llm, backend, question, layer, value_index, max_retries=2):
    """Ablation rung: semantic + value-linking only (no plan, no self-consistency, no critic)."""
    return _grounded(
        llm,
        backend,
        question,
        layer,
        value_index,
        use_plan=False,
        use_critic=False,
        k=1,
        max_retries=max_retries,
    )


def run_plan_sc(llm, backend, question, layer, value_index, k_candidates=3, max_retries=2):
    """Ablation rung: + Planner decomposition + K-candidate self-consistency (critic off)."""
    return _grounded(
        llm,
        backend,
        question,
        layer,
        value_index,
        use_plan=True,
        use_critic=False,
        k=k_candidates,
        max_retries=max_retries,
    )


def run_grounded(llm, backend, question, layer, value_index, k_candidates=3, max_retries=2):
    """Ablation rung: full grounded path (+ reflexion critic)."""
    return _grounded(
        llm,
        backend,
        question,
        layer,
        value_index,
        use_plan=True,
        use_critic=True,
        k=k_candidates,
        max_retries=max_retries,
    )
