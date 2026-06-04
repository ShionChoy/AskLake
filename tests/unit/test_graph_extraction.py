from engine.graph.extraction import PlotDoc, build_graph, extract_triples
from engine.graph.ontology import GraphOntology
from engine.llm.fake import FakeLLMProvider

_ONT = GraphOntology(
    entity_types=("Film", "Person", "Theme"),
    relation_types=("HAS_THEME", "DIRECTED_BY"),
    hint="Extract director and themes.",
)

# Note the third line uses a relation NOT in the ontology -> must be dropped.
_RESPONSE = (
    "Inception | DIRECTED_BY | Christopher Nolan\n"
    "Inception | HAS_THEME | dreams\n"
    "Inception | SET_IN | a dream\n"
    "garbage line without delimiters\n"
)


def test_extract_triples_parses_and_filters_by_ontology():
    llm = FakeLLMProvider(responses=[_RESPONSE])
    doc = PlotDoc(id="plot_inception", title="Inception", text="A thief enters dreams.")
    triples = extract_triples(llm, doc, _ONT)
    rels = {t.relation for t in triples}
    assert rels == {"DIRECTED_BY", "HAS_THEME"}  # SET_IN dropped (not in ontology)
    assert all(t.source == "plot_inception" for t in triples)  # citation attached
    # the ontology's allowed relations are surfaced to the model
    assert "HAS_THEME" in llm.prompts[0] and "DIRECTED_BY" in llm.prompts[0]


def test_build_graph_loads_all_docs_in_order():
    llm = FakeLLMProvider(
        responses=[
            "A | HAS_THEME | x\n",
            "B | HAS_THEME | y\n",
        ]
    )
    docs = [
        PlotDoc("doc_a", "A", "..."),
        PlotDoc("doc_b", "B", "..."),
    ]
    store = build_graph(llm, docs, _ONT)
    sources = {t.source for t in store.triples()}
    assert sources == {"doc_a", "doc_b"}
    assert len(store.triples()) == 2
