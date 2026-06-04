from engine.graph.retriever import GraphRetriever, RetrievedSubgraph
from engine.graph.store import InMemoryGraphStore
from engine.ports.graph_store import Triple


def _store() -> InMemoryGraphStore:
    g = InMemoryGraphStore()
    g.add(Triple("Inception", "DIRECTED_BY", "Christopher Nolan", "plot_inception"))
    g.add(Triple("Inception", "HAS_THEME", "dreams", "plot_inception"))
    g.add(Triple("The Dark Knight", "DIRECTED_BY", "Christopher Nolan", "plot_tdk"))
    g.add(Triple("The Dark Knight", "HAS_THEME", "chaos", "plot_tdk"))
    g.add(Triple("Titanic", "HAS_THEME", "romance", "plot_titanic"))
    return g


def test_seeds_match_entities_named_in_question():
    ctx = GraphRetriever(_store(), max_hops=1).retrieve("themes in Inception")
    assert ctx.seeds == ("Inception",)


def test_multihop_reaches_themes_via_director():
    # seed = Christopher Nolan; hop 1 -> his films; hop 2 -> their themes
    ctx = GraphRetriever(_store(), max_hops=2).retrieve("films by Christopher Nolan")
    objs = {t.obj for t in ctx.triples}
    assert "dreams" in objs and "chaos" in objs  # reached through the films
    assert "romance" not in objs  # Titanic is unconnected to Nolan


def test_no_seed_returns_empty_subgraph():
    ctx = GraphRetriever(_store()).retrieve("how many films are there")
    assert isinstance(ctx, RetrievedSubgraph)
    assert ctx.seeds == () and ctx.triples == ()
