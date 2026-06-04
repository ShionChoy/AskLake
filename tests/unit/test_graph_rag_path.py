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
