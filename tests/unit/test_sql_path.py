from engine.lakehouse.duckdb_backend import DuckDBBackend
from engine.llm.fake import FakeLLMProvider
from engine.ports.retrieval import RetrievalPath
from engine.ports.storage import QueryResult
from engine.retrieval.sql_path import SqlPath, infer_chart_spec
from engine.semantic.raw_schema import RawSchemaProvider

SEED = "CREATE TABLE m AS SELECT * FROM (VALUES ('A', 8.8), ('B', 7.0)) t(title, rating);"


def _make_path(sql_response):
    backend = DuckDBBackend()
    backend.setup(SEED)
    llm = FakeLLMProvider(responses=[sql_response])
    return SqlPath(llm, RawSchemaProvider(backend), backend)


def test_sql_path_is_a_retrieval_path():
    assert isinstance(_make_path("SELECT 1"), RetrievalPath)


def test_sql_path_runs_query_and_infers_bar_chart():
    rr = _make_path("SELECT title, rating FROM m ORDER BY rating DESC").run("top movies")
    assert rr.path == "sql"
    assert rr.sql.startswith("SELECT")
    assert rr.result.rows[0] == ("A", 8.8)
    assert rr.chart_spec == {"type": "bar", "x": "title", "y": "rating"}


def test_sql_path_degrades_gracefully_on_bad_sql():
    rr = _make_path("SELECT * FROM does_not_exist").run("bad")
    assert rr.result is None
    assert "failed" in rr.narrative.lower()


def test_infer_chart_spec_none_when_no_numeric_measure():
    assert infer_chart_spec(QueryResult(["a", "b"], [("x", "y")])) is None


def test_sql_path_handles_empty_sql():
    rr = _make_path("").run("nothing")
    assert rr.result is None
    assert "no query" in rr.narrative.lower()


def test_infer_chart_spec_none_for_bool_measure():
    assert infer_chart_spec(QueryResult(["a", "b"], [("x", True)])) is None
