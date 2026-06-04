"""Phase 4 demo: GraphRAG 2nd retrieval path + Router fusion, hermetic (no API key, no Neo4j).

The knowledge graph is built by the real LLM extraction adapter driven by FakeLLMProvider over a
tiny SYNTHETIC plot corpus (not raw CMU text). The Router scores each question and dispatches to
SqlPath, GraphRagPath, or fuses both via the Synthesizer. The fusion question returns the SQL film
table plus graph themes with traceable plot citations."""

from __future__ import annotations

from engine.graph.extraction import PlotDoc, build_graph
from engine.graph.ontology import GraphOntology
from engine.lakehouse.duckdb_backend import DuckDBBackend
from engine.llm.fake import FakeLLMProvider
from engine.retrieval.graph_rag_path import GraphRagPath
from engine.retrieval.router import Router
from engine.retrieval.sql_path import SqlPath
from engine.semantic.raw_schema import RawSchemaProvider

# --- structured side: a few pre-2013 Christopher Nolan films + ratings -------------------------
SEED = (
    "CREATE TABLE films AS SELECT * FROM (VALUES "
    "('Inception', 8.8, 2010), ('The Dark Knight', 9.0, 2008), ('Interstellar', 8.6, 2014)"
    ") t(title, averageRating, startYear);"
)
_FILM_SQL = (
    "SELECT title, averageRating FROM films WHERE startYear < 2013 "
    "ORDER BY averageRating DESC LIMIT 5"
)

# --- unstructured side: synthetic plot blurbs (NOT raw CMU) for the hermetic extraction ---------
_ONTOLOGY = GraphOntology(
    entity_types=("Film", "Person", "Theme"),
    relation_types=("HAS_THEME", "DIRECTED_BY"),
    hint="Extract the director and the central themes.",
)
_DOCS = [
    PlotDoc("plot_inception", "Inception", "A thief who steals secrets from dreams..."),
    PlotDoc("plot_tdk", "The Dark Knight", "A vigilante confronts an agent of chaos..."),
]
# Canned extraction output, one block per doc (FakeLLMProvider cycles in order). Both films share
# the theme 'identity', which the graph connects through their common director.
_EXTRACTION = [
    (
        "Inception | DIRECTED_BY | Christopher Nolan\n"
        "Inception | HAS_THEME | dreams\n"
        "Inception | HAS_THEME | identity\n"
    ),
    (
        "The Dark Knight | DIRECTED_BY | Christopher Nolan\n"
        "The Dark Knight | HAS_THEME | chaos\n"
        "The Dark Knight | HAS_THEME | identity\n"
    ),
]

_SQL_QUESTION = "How many films are rated above 8?"
_GRAPH_QUESTION = "What are the common themes in these plots?"
_FUSION_QUESTION = (
    "Christopher Nolan's highest-rated films before 2013 and their common plot themes"
)


def _build_router() -> Router:
    backend = DuckDBBackend()
    backend.setup(SEED)
    sql_path = SqlPath(FakeLLMProvider(responses=[_FILM_SQL]), RawSchemaProvider(backend), backend)
    store = build_graph(FakeLLMProvider(responses=_EXTRACTION), _DOCS, _ONTOLOGY)
    graph_path = GraphRagPath(store)
    return Router(sql_path, graph_path, entity_vocab=store.entities())


def run_demo_p4() -> dict:
    router = _build_router()

    def decide(q: str) -> str:
        d = router.route(q)
        return "+".join(d.paths) if d.fuse else d.paths[0]

    routes = {
        "sql": decide(_SQL_QUESTION),
        "graph": decide(_GRAPH_QUESTION),
        "fusion": decide(_FUSION_QUESTION),
    }
    rr = router.run(_FUSION_QUESTION)
    fused = {
        "path": rr.path,
        "sql_rows": [list(r) for r in rr.result.rows] if rr.result else None,
        "narrative": rr.narrative,
    }
    return {"routes": routes, "fused": fused}


if __name__ == "__main__":
    out = run_demo_p4()
    print("routes:", out["routes"])
    print("fused path:", out["fused"]["path"])
    print("fused SQL rows:", out["fused"]["sql_rows"])
    print("fused narrative:")
    print(out["fused"]["narrative"])
    print("demo-p4 OK")
