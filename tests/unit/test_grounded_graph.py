from engine.agents.grounded_graph import build_grounded_sql_graph
from engine.lakehouse.duckdb_backend import DuckDBBackend
from engine.llm.fake import FakeLLMProvider
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


def test_grounded_graph_happy_path_single_candidate():
    backend = _backend()
    provider = SemanticLayerProvider(_layer())
    llm = FakeLLMProvider(
        responses=["SELECT title, averageRating FROM films ORDER BY averageRating DESC LIMIT 5"]
    )
    graph = build_grounded_sql_graph(
        llm,
        provider,
        backend.run_sql,
        value_index=build_value_index(_layer(), backend),
        k_candidates=1,
    )
    state = graph.invoke({"question": "top 5 highest rated sci fi films"})
    assert state["sql"].upper().startswith("SELECT")
    assert state["result"].rows[0][0] == "A"
    assert not state.get("error")
    # value hint reached the writer prompt
    assert "genres LIKE '%Sci-Fi%'" in llm.prompts[-1] or "genres LIKE '%Sci-Fi%'" in llm.prompts[0]


def test_grounded_graph_critic_triggers_correction_on_missing_limit():
    backend = _backend()
    provider = SemanticLayerProvider(_layer())
    llm = FakeLLMProvider(
        responses=[
            "SELECT title, averageRating FROM films ORDER BY averageRating DESC",  # no LIMIT -> critic rejects  # noqa: E501
            "SELECT title, averageRating FROM films ORDER BY averageRating DESC LIMIT 5",  # corrected  # noqa: E501
        ]
    )
    # use_plan=False so the 2-response FakeLLM maps exactly to write + correct
    # (the plan node would otherwise consume one response and shift the sequence).
    graph = build_grounded_sql_graph(
        llm, provider, backend.run_sql, k_candidates=1, max_retries=2, use_plan=False
    )
    state = graph.invoke({"question": "top 5 highest rated movies"})
    assert state["attempts"] == 1
    assert "LIMIT" in state["sql"].upper()


def test_grounded_graph_use_critic_false_skips_correctness_loop():
    backend = _backend()
    provider = SemanticLayerProvider(_layer())
    llm = FakeLLMProvider(
        responses=["SELECT title, averageRating FROM films ORDER BY averageRating DESC"]
    )
    graph = build_grounded_sql_graph(
        llm, provider, backend.run_sql, k_candidates=1, use_critic=False, max_retries=2
    )
    state = graph.invoke({"question": "top 5 highest rated movies"})
    assert state["attempts"] == 0  # no critic-driven correction; runs-but-imperfect is accepted
