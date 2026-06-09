import pytest

from eval.hermetic import run_hermetic_comparison


def test_agentic_beats_baseline_on_mini_set():
    baseline, agentic = run_hermetic_comparison()
    assert baseline.n == agentic.n == 3
    assert agentic.execution_accuracy > baseline.execution_accuracy
    assert agentic.execution_accuracy == 1.0
    assert baseline.execution_accuracy == pytest.approx(2 / 3)
    assert agentic.avg_attempts > 0  # at least one self-correction happened


def test_ablation_runners_return_sql_and_attempts():
    from engine.lakehouse.duckdb_backend import DuckDBBackend
    from engine.llm.fake import FakeLLMProvider
    from engine.semantic.semantic_model import ColumnDef, SemanticLayer, TableDef
    from engine.semantic.value_index import build_value_index
    from eval.systems import run_grounded, run_plan_sc, run_value_link

    backend = DuckDBBackend()
    backend.setup(
        "CREATE TABLE films AS SELECT * FROM (VALUES "
        "('A', 9.0, 'Sci-Fi'), ('B', 7.0, 'Comedy')) t(title, averageRating, genres);"
    )
    layer = SemanticLayer(
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
    vidx = build_value_index(layer, backend)
    good = "SELECT title, averageRating FROM films ORDER BY averageRating DESC LIMIT 5"
    llm = FakeLLMProvider(responses=[good])

    for runner in (run_value_link, run_plan_sc, run_grounded):
        sql, attempts = runner(llm, backend, "top 5 highest rated sci fi films", layer, vidx)
        assert sql.upper().startswith("SELECT")
        assert isinstance(attempts, int)
