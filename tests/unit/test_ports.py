from engine.ports.governance import GovernanceHook
from engine.ports.llm import LLMProvider
from engine.ports.retrieval import RetrievalResult
from engine.ports.storage import ColumnSchema, QueryResult, TableSchema


def test_query_result_shape():
    qr = QueryResult(columns=["a"], rows=[(1,)])
    assert qr.columns == ["a"]
    assert qr.rows == [(1,)]


def test_table_schema_shape():
    ts = TableSchema(name="t", columns=[ColumnSchema(name="c", type="BIGINT")])
    assert ts.columns[0].name == "c"


def test_protocols_are_runtime_checkable():
    class DummyLLM:
        def complete(self, prompt, system=None):
            return "x"

    class DummyGov:
        def before_query(self, sql, role):
            return sql

        def after_result(self, result, role):
            return result

    assert isinstance(DummyLLM(), LLMProvider)
    assert isinstance(DummyGov(), GovernanceHook)


def test_retrieval_result_shape():
    rr = RetrievalResult(
        path="sql",
        sql="SELECT 1",
        result=QueryResult(["x"], [(1,)]),
        narrative="ok",
        chart_spec=None,
    )
    assert rr.path == "sql"
