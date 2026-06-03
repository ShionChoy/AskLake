from engine.agents.graph import build_sql_graph, extract_sql
from engine.llm.fake import FakeLLMProvider


class StubSchema:
    def schema_context(self, question: str) -> str:
        return "title_basics(tconst TEXT, primaryTitle TEXT)"


def test_extract_sql_strips_code_fences_and_semicolons():
    assert extract_sql("```sql\nSELECT 1;\n```") == "SELECT 1"
    assert extract_sql("```\nSELECT 2\n```") == "SELECT 2"
    assert extract_sql("SELECT 3") == "SELECT 3"


def test_graph_generates_sql_and_grounds_on_schema():
    llm = FakeLLMProvider(responses=["```sql\nSELECT 1\n```"])
    graph = build_sql_graph(llm, StubSchema())
    state = graph.invoke({"question": "how many titles?"})
    assert state["sql"] == "SELECT 1"
    assert "title_basics" in llm.prompts[0]  # schema injected into the prompt
