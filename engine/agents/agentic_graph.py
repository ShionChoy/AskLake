from __future__ import annotations

from collections.abc import Callable

from langgraph.graph import END, START, StateGraph

from engine.agents.graph import PROMPT_TEMPLATE, SYSTEM_PROMPT, extract_sql
from engine.ports.agentgraph import GraphState
from engine.ports.llm import LLMProvider
from engine.ports.schema import SchemaProvider
from engine.ports.storage import QueryResult

CORRECTION_SYSTEM = (
    "You are a senior analytics engineer fixing a failed DuckDB SQL query. "
    "Use only the listed tables/columns. Return only the corrected SQL, no prose."
)

CORRECTION_TEMPLATE = """The previous SQL query failed.

Database schema:
{schema}

Question: {question}

Previous SQL:
{sql}

Error:
{error}

Write ONE corrected DuckDB SQL query. Return only the SQL."""


def build_agentic_sql_graph(
    llm: LLMProvider,
    schema_provider: SchemaProvider,
    executor: Callable[[str], QueryResult],
    max_retries: int = 2,
):
    """Cyclic self-correcting graph (P2), additive sibling of build_sql_graph (P1, the eval
    baseline).

    SQLWriter -> Validator -(error & attempts<max_retries)-> SelfCorrect -> Validator ... -> END.
    `executor(sql)` runs the SQL (raising on error); callers inject one that wraps governance +
    StorageBackend, so this graph imports no storage/governance code. Termination is bounded by the
    explicit `attempts` counter (writer + at most `max_retries` corrections).
    """

    def write_sql(state: GraphState) -> GraphState:
        question = state["question"]
        ctx = schema_provider.schema_context(question)
        prompt = PROMPT_TEMPLATE.format(schema=ctx, question=question)
        raw = llm.complete(prompt, system=SYSTEM_PROMPT)
        return {"schema_context": ctx, "sql": extract_sql(raw), "attempts": 0}

    def validate(state: GraphState) -> GraphState:
        try:
            result = executor(state["sql"])
        except Exception as exc:  # noqa: BLE001 - the error feeds the self-correct loop
            return {"error": str(exc)}
        return {"result": result, "error": ""}

    def self_correct(state: GraphState) -> GraphState:
        prompt = CORRECTION_TEMPLATE.format(
            schema=state.get("schema_context", ""),
            question=state["question"],
            sql=state.get("sql", ""),
            error=state.get("error", ""),
        )
        raw = llm.complete(prompt, system=CORRECTION_SYSTEM)
        return {"sql": extract_sql(raw), "attempts": state.get("attempts", 0) + 1}

    def route(state: GraphState) -> str:
        if state.get("error") and state.get("attempts", 0) < max_retries:
            return "retry"
        return "done"

    graph = StateGraph(GraphState)
    graph.add_node("write_sql", write_sql)
    graph.add_node("validate", validate)
    graph.add_node("self_correct", self_correct)
    graph.add_edge(START, "write_sql")
    graph.add_edge("write_sql", "validate")
    graph.add_conditional_edges("validate", route, {"retry": "self_correct", "done": END})
    graph.add_edge("self_correct", "validate")
    return graph.compile()
