from engine.lakehouse.duckdb_backend import DuckDBBackend
from engine.semantic.semantic_model import ColumnDef, SemanticLayer, TableDef
from engine.semantic.value_index import ValueIndex, build_value_index, format_hints


def _layer():
    return SemanticLayer(
        tables=(
            TableDef(
                name="films",
                columns=(
                    ColumnDef("title"),
                    ColumnDef("genres", link="categorical"),
                    ColumnDef("lead", link="entity"),
                ),
            ),
        )
    )


def _backend():
    b = DuckDBBackend()
    b.setup(
        "CREATE TABLE films AS SELECT * FROM (VALUES "
        "('A','Sci-Fi,Drama','Keanu Reeves'), "
        "('B','Comedy','Bill Murray')) t(title, genres, lead);"
    )
    return b


def test_categorical_token_subset_match_emits_like_hint():
    idx = build_value_index(_layer(), _backend())
    hints = idx.link("highest rated sci fi films")
    cat = [h for h in hints if h.mode == "categorical"]
    assert any(h.column == "genres" and h.value == "Sci-Fi" for h in cat)
    assert "genres LIKE '%Sci-Fi%'" in format_hints(hints)


def test_entity_probe_resolves_canonical_spelling():
    idx = build_value_index(_layer(), _backend())
    hints = idx.link("movies with Keanu Reeves")
    ent = [h for h in hints if h.mode == "entity"]
    assert any(h.column == "lead" and h.value == "Keanu Reeves" for h in ent)
    assert "lead = 'Keanu Reeves'" in format_hints(hints)


def test_no_spurious_hint_when_nothing_matches():
    idx = build_value_index(_layer(), _backend())
    assert idx.link("how many titles are there") == []


def test_empty_index_links_nothing_gracefully():
    assert ValueIndex().link("anything") == []
