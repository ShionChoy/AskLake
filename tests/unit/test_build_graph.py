from engine.graph.extraction import PlotDoc
from engine.graph.ontology import GraphOntology
from engine.graph.persistence import load_store
from engine.llm.fake import FakeLLMProvider
from scripts.build_graph import build_and_save


def test_build_and_save_writes_triples(tmp_path):
    docs = [
        PlotDoc(id="cmu:1", title="The Dark Knight", text="Batman vs the Joker."),
        PlotDoc(id="cmu:2", title="Inception", text="A thief in dreams."),
    ]
    ontology = GraphOntology(
        entity_types=("Film", "Person", "Theme"),
        relation_types=("HAS_THEME", "DIRECTED_BY"),
        hint="Extract director and themes.",
    )
    extraction = [
        "The Dark Knight | DIRECTED_BY | Christopher Nolan\nThe Dark Knight | HAS_THEME | chaos\n",
        "Inception | DIRECTED_BY | Christopher Nolan\nInception | HAS_THEME | dreams\n",
    ]
    out = tmp_path / "triples.jsonl"
    count = build_and_save(docs, FakeLLMProvider(extraction), ontology, out)
    assert count == 4
    store = load_store(out)
    assert "Christopher Nolan" in store.entities()
    rows = [(t.subject, t.relation, t.obj, t.source) for t in store.triples()]
    assert ("The Dark Knight", "HAS_THEME", "chaos", "cmu:1") in rows
