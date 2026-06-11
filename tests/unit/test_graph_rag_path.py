from engine.graph.store import InMemoryGraphStore
from engine.ports.graph_store import Triple
from engine.ports.retrieval import RetrievalPath
from engine.retrieval.graph_rag_path import GraphRagPath


def _path() -> GraphRagPath:
    g = InMemoryGraphStore()
    g.add(Triple("Inception", "HAS_THEME", "dreams", "plot_inception"))
    g.add(Triple("Inception", "DIRECTED_BY", "Christopher Nolan", "plot_inception"))
    return GraphRagPath(g)


def test_is_retrieval_path():
    p = _path()
    assert isinstance(p, RetrievalPath)
    assert p.name == "graph"


def test_can_handle_theme_question_and_named_entity():
    p = _path()
    assert p.can_handle("what are the common themes here") is True  # graph hint word
    assert p.can_handle("tell me about Inception") is True  # names a known entity
    assert p.can_handle("how many rows are there") is False  # neither


def test_run_returns_cited_facts():
    rr = _path().run("themes in Inception")
    assert rr.path == "graph"
    assert rr.sql is None
    assert rr.result is not None
    assert rr.result.columns == ["subject", "relation", "object", "source"]
    assert ("Inception", "HAS_THEME", "dreams", "plot_inception") in rr.result.rows
    assert "[plot_inception]" in rr.narrative  # citation surfaced in the narrative


def test_run_with_no_match_degrades_gracefully():
    rr = _path().run("unrelated question about nothing")
    assert rr.path == "graph"
    assert rr.result is None
    assert "No matching facts" in rr.narrative


def test_run_caps_rows_and_notes_truncation():
    g = InMemoryGraphStore()
    for i in range(50):
        g.add(Triple("Hub", "REL", f"Leaf {i}", f"s{i}"))  # seed "Hub" -> 50 facts
    rr = GraphRagPath(g, max_rows=10).run("hub")
    assert rr.result is not None
    assert len(rr.result.rows) == 10  # capped
    assert "showing first 10 of" in rr.narrative  # truncation surfaced
    assert rr.narrative.count("[s") == 10  # only the shown facts are cited


def test_empty_hint_is_configurable():
    g = InMemoryGraphStore()
    g.add(Triple("Inception", "HAS_THEME", "dreams", "s"))
    # default keeps the generic message
    assert "No matching facts" in GraphRagPath(g).run("totally unrelated zzz").narrative
    # an injected hint replaces it on the no-match case
    rr = GraphRagPath(g, empty_hint="not in the graph; use SQL").run("totally unrelated zzz")
    assert rr.result is None
    assert rr.narrative == "not in the graph; use SQL"


def test_typed_path_cast_vs_theme_differ():
    from engine.graph.intent import IntentResolver
    from engine.graph.ontology import GraphOntology, Intent

    g = InMemoryGraphStore()
    g.add(Triple("Inception", "ACTED_IN", "Leonardo DiCaprio", "s"))
    g.add(Triple("Inception", "HAS_THEME", "dreams", "s"))
    ont = GraphOntology(
        relation_types=("ACTED_IN", "HAS_THEME"),
        connective_relations=("HAS_THEME",),
        intents=(
            Intent("cast", frozenset({"actors"}), frozenset({"ACTED_IN"}), "entity_lookup"),
            Intent("themes", frozenset({"themes"}), frozenset({"HAS_THEME"}), "cluster"),
        ),
    )
    p = GraphRagPath(
        g, connective_relations=frozenset({"HAS_THEME"}), intent_resolver=IntentResolver(ont)
    )
    cast = {r[1] for r in p.run("actors of Inception").result.rows}
    themes = {r[1] for r in p.run("themes of Inception").result.rows}
    assert cast == {"ACTED_IN"} and themes == {"HAS_THEME"}
