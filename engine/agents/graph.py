from __future__ import annotations

import re

from langgraph.graph import END, START, StateGraph

from engine.ports.agentgraph import GraphState
from engine.ports.llm import LLMProvider
from engine.ports.schema import SchemaProvider

SYSTEM_PROMPT = (
    "You are a senior analytics engineer. Given a database schema and a question, "
    "write ONE DuckDB SQL query that answers it. Use only the listed tables/columns. "
    "Return only the SQL, no prose."
)

PROMPT_TEMPLATE = """Database schema:
{schema}

Question: {question}

Return only one DuckDB SQL query."""


def extract_sql(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1).strip()
    return text.rstrip(";").strip()


def build_sql_graph(llm: LLMProvider, schema_provider: SchemaProvider):
    """Single-node graph (P1). P2 adds Planner/Validator/SelfCorrect nodes additively."""

    def generate_sql(state: GraphState) -> GraphState:
        question = state["question"]
        ctx = schema_provider.schema_context(question)
        prompt = PROMPT_TEMPLATE.format(schema=ctx, question=question)
        raw = llm.complete(prompt, system=SYSTEM_PROMPT)
        return {"schema_context": ctx, "sql": extract_sql(raw)}

    graph = StateGraph(GraphState)
    graph.add_node("generate_sql", generate_sql)
    graph.add_edge(START, "generate_sql")
    graph.add_edge("generate_sql", END)
    return graph.compile()
