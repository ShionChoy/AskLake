"""Phase 6 demo: grounded, correctness-aware NL->SQL, hermetic (no API key).

A hard top-N question over a tiny films table. The Planner runs, then three FakeLLM
candidates are produced; one omits the genre filter (wrong rows) and is voted out by
self-consistency; the grounded + consistent answer passes the critic with zero
corrections."""

from __future__ import annotations

from engine.lakehouse.duckdb_backend import DuckDBBackend
from engine.llm.fake import FakeLLMProvider
from engine.retrieval.grounded_sql_path import GroundedSqlPath
from engine.semantic.semantic_layer import SemanticLayerProvider
from engine.semantic.semantic_model import ColumnDef, SemanticLayer, TableDef
from engine.semantic.value_index import build_value_index

SEED = (
    "CREATE TABLE films AS SELECT * FROM (VALUES "
    "('Inception', 8.8, 'Sci-Fi'), ('Interstellar', 8.6, 'Sci-Fi'), "
    "('Dumb and Dumber', 7.3, 'Comedy')) t(title, averageRating, genres);"
)

_PLAN = "1. filter films to genres LIKE '%Sci-Fi%'\n2. order by averageRating desc\n3. limit 5"
_GOOD = (
    "SELECT title, averageRating FROM films WHERE genres LIKE '%Sci-Fi%'"
    " ORDER BY averageRating DESC LIMIT 5"
)
# _BAD omits the genre filter: it executes fine but returns the wrong rows (Comedy included).
_BAD = "SELECT title, averageRating FROM films ORDER BY averageRating DESC LIMIT 5"


def _layer() -> SemanticLayer:
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


def run_demo_p6() -> dict:
    backend = DuckDBBackend()
    backend.setup(SEED)
    layer = _layer()
    # difficulty=hard -> plan(1 call) then write(K=3 calls): plan + BAD + GOOD + GOOD.
    llm = FakeLLMProvider(responses=[_PLAN, _BAD, _GOOD, _GOOD])  # majority candidate = GOOD
    path = GroundedSqlPath(
        llm,
        SemanticLayerProvider(layer),
        backend,
        value_index=build_value_index(layer, backend),
        k_candidates=3,
    )
    rr = path.run("top 5 highest rated sci fi films")
    return {
        "sql": rr.sql,
        "rows": [list(r) for r in rr.result.rows] if rr.result else None,
        "narrative": rr.narrative,
        "discarded_minority": rr.sql == _GOOD,  # GOOD won the vote over BAD
        "corrections": int(rr.narrative.split("after ")[1].split(" correction")[0])
        if rr.result
        else -1,
    }


if __name__ == "__main__":
    out = run_demo_p6()
    print("chosen sql:", out["sql"])
    print("rows:", out["rows"])
    print("narrative:", out["narrative"])
    print("minority discarded:", out["discarded_minority"])
    print("demo-p6 OK")
