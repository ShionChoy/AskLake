"""Regression coverage for the baseline and grounded agent graphs."""

import inspect

import engine.agents.agentic_graph as ag
import engine.agents.graph as base
from engine.agents.agentic_graph import build_agentic_sql_graph
from engine.llm.fake import FakeLLMProvider
from engine.ports.storage import QueryResult


class _StubSchema:
    def schema_context(self, question: str) -> str:
        return "movies(title TEXT, averageRating DOUBLE)"


def test_agentic_path_behavior_unchanged():
    def executor(sql: str) -> QueryResult:
        if "averageRating" not in sql:
            raise ValueError('Referenced column "rating" not found')
        return QueryResult(["title", "averageRating"], [("A", 8.8)])

    llm = FakeLLMProvider(
        responses=[
            "SELECT title, rating FROM movies",
            "SELECT title, averageRating FROM movies",
        ]
    )
    graph = build_agentic_sql_graph(llm, _StubSchema(), executor, max_retries=2)
    state = graph.invoke({"question": "q"})
    assert state["attempts"] == 1 and state["result"].rows == [("A", 8.8)]


def test_existing_agent_modules_do_not_depend_on_new_code():
    # The baseline builders must not import the newer grounded stack.
    for mod in (base, ag):
        src = inspect.getsource(mod)
        assert "grounded_graph" not in src
        assert "value_index" not in src
        assert "critic" not in src
