from engine.lakehouse.duckdb_backend import DuckDBBackend
from engine.llm.fake import FakeLLMProvider
from engine.ports.retrieval import RetrievalPath
from engine.retrieval.grounded_sql_path import GroundedSqlPath
from engine.semantic.semantic_layer import SemanticLayerProvider
from engine.semantic.semantic_model import ColumnDef, SemanticLayer, TableDef
from engine.semantic.value_index import build_value_index


def _backend():
    b = DuckDBBackend()
    b.setup(
        "CREATE TABLE films AS SELECT * FROM (VALUES "
        "('A', 9.0, 'Sci-Fi'), ('B', 7.0, 'Comedy')) t(title, averageRating, genres);"
    )
    return b


def _layer():
    return SemanticLayer(
        tables=(
            TableDef(
                name="films",
                columns=(
                    ColumnDef("title"),
                    ColumnDef("averageRating", type="DOUBLE"),
                    ColumnDef("genres", link="categorical"),
                ),
            ),
        )
    )


def test_grounded_sql_path_runs_and_is_a_retrieval_path():
    backend = _backend()
    llm = FakeLLMProvider(
        responses=["SELECT title, averageRating FROM films ORDER BY averageRating DESC LIMIT 5"]
    )
    path = GroundedSqlPath(
        llm,
        SemanticLayerProvider(_layer()),
        backend,
        value_index=build_value_index(_layer(), backend),
        k_candidates=1,
    )
    assert isinstance(path, RetrievalPath)
    assert path.name == "sql"
    rr = path.run("top 5 highest rated sci fi films")
    assert rr.path == "sql"
    assert rr.result.rows[0][0] == "A"
    assert rr.chart_spec and rr.chart_spec["type"] == "bar"
