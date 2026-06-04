from engine.agents.agentic_graph import build_agentic_sql_graph
from engine.llm.fake import FakeLLMProvider
from engine.ports.storage import QueryResult


class StubSchema:
    def schema_context(self, question: str) -> str:
        return "movies(title TEXT, averageRating DOUBLE)"


def test_graph_self_corrects_after_execution_error():
    def executor(sql: str) -> QueryResult:
        if "averageRating" not in sql:
            raise ValueError('Referenced column "rating" not found')
        return QueryResult(["title", "averageRating"], [("A", 8.8)])

    llm = FakeLLMProvider(
        responses=[
            "SELECT title, rating FROM movies",  # bad column -> executor raises
            "SELECT title, averageRating FROM movies",  # corrected
        ]
    )
    graph = build_agentic_sql_graph(llm, StubSchema(), executor, max_retries=2)
    state = graph.invoke({"question": "ratings?"})

    assert state["attempts"] == 1
    res = state.get("result")
    assert res is not None and res.rows == [("A", 8.8)]
    assert not state.get("error")
    # schema grounded into the first (writer) prompt
    assert "movies(" in llm.prompts[0]


def test_graph_terminates_after_max_retries():
    def executor(sql: str) -> QueryResult:
        raise ValueError("always fails")

    llm = FakeLLMProvider(responses=["SELECT 1"])  # cycles the same bad sql
    graph = build_agentic_sql_graph(llm, StubSchema(), executor, max_retries=2)
    state = graph.invoke({"question": "q"})

    assert state["attempts"] == 2  # writer + 2 corrections, then stop (bounded)
    assert state.get("result") is None
    assert state["error"]
