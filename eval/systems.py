from __future__ import annotations

from engine.agents.agentic_graph import build_agentic_sql_graph
from engine.agents.graph import build_sql_graph
from engine.ports.llm import LLMProvider
from engine.ports.storage import QueryResult, StorageBackend
from engine.semantic.raw_schema import RawSchemaProvider


def run_baseline(llm: LLMProvider, backend: StorageBackend, question: str) -> tuple[str, int]:
    """Naive single-prompt baseline = P1 single-node graph: one shot, no self-correction."""
    graph = build_sql_graph(llm, RawSchemaProvider(backend))
    state = graph.invoke({"question": question})
    return state.get("sql", ""), 0


def run_agentic(
    llm: LLMProvider, backend: StorageBackend, question: str, max_retries: int = 2
) -> tuple[str, int]:
    """Agentic self-correcting system (P2): returns the final SQL and the number of corrections."""

    def executor(sql: str) -> QueryResult:
        return backend.run_sql(sql)

    graph = build_agentic_sql_graph(llm, RawSchemaProvider(backend), executor, max_retries)
    state = graph.invoke({"question": question})
    return state.get("sql", ""), state.get("attempts", 0)
