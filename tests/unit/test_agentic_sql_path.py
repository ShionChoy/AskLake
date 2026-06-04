from engine.lakehouse.duckdb_backend import DuckDBBackend
from engine.llm.fake import FakeLLMProvider
from engine.ports.retrieval import RetrievalPath
from engine.retrieval.agentic_sql_path import AgenticSqlPath
from engine.semantic.raw_schema import RawSchemaProvider

SEED = (
    "CREATE TABLE movies AS SELECT * FROM (VALUES ('A', 8.8), ('B', 7.0)) t(title, averageRating);"
)


def _make(responses):
    backend = DuckDBBackend()
    backend.setup(SEED)
    return AgenticSqlPath(
        FakeLLMProvider(responses=responses), RawSchemaProvider(backend), backend, max_retries=2
    )


def test_is_a_retrieval_path():
    assert isinstance(_make(["SELECT 1"]), RetrievalPath)


def test_self_corrects_bad_column_then_succeeds():
    rr = _make(
        [
            "SELECT title, rating FROM movies ORDER BY rating DESC",  # bad column
            "SELECT title, averageRating FROM movies ORDER BY averageRating DESC",  # corrected
        ]
    ).run("top movies")
    assert rr.result is not None
    assert rr.result.rows[0] == ("A", 8.8)
    assert "self-correction" in rr.narrative
    assert rr.chart_spec == {"type": "bar", "x": "title", "y": "averageRating"}


def test_degrades_after_max_retries():
    rr = _make(["SELECT * FROM does_not_exist"]).run("bad")
    assert rr.result is None
    assert "failed" in rr.narrative.lower()
