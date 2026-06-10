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


def _hub_store(n_films: int = 500) -> InMemoryGraphStore:
    g = InMemoryGraphStore()
    for i in range(n_films):
        g.add(Triple(f"Film {i}", "HAS_GENRE", "Drama", f"src{i}"))
    # a second-hop theme reachable ONLY by expanding a film off the hub
    g.add(Triple("Film 0", "HAS_THEME", "revenge", "src0"))
    return g


def _wide_store() -> InMemoryGraphStore:
    g = InMemoryGraphStore()
    for i in range(40):  # Hub -> 40 mids (Hub degree 40, expandable)
        g.add(Triple("Hub", "REL", f"Mid {i}", "s"))
        for j in range(40):  # each mid -> 40 leaves (mid degree 41, expandable)
            g.add(Triple(f"Mid {i}", "REL", f"Leaf {i}-{j}", "s"))
    return g


def test_retrieve_capped_at_max_triples():
    r = GraphRetriever(
        _wide_store(), max_hops=2, max_triples=300, max_neighbors_per_node=50, max_degree=100
    )
    ctx = r.retrieve("hub")
    assert len(ctx.triples) <= 300  # budget enforced
    assert len(ctx.triples) > 50  # multi-hop actually happened (not just hop 1)


def test_hub_node_is_not_expanded():
    r = GraphRetriever(
        _hub_store(500), max_hops=2, max_triples=300, max_neighbors_per_node=50, max_degree=100
    )
    ctx = r.retrieve("drama")  # seeds the degree-500 "Drama" hub
    assert len(ctx.triples) <= 50  # per-node fan-out cap: only 50 of 500 edges
    assert all(t.obj != "revenge" for t in ctx.triples)  # hub not expanded -> no 2nd hop


def _attr_store() -> InMemoryGraphStore:
    g = InMemoryGraphStore()
    g.add(Triple("Drama Queen", "HAS_GENRE", "Drama", "s1"))  # film whose TITLE contains "drama"
    g.add(Triple("Inception", "HAS_GENRE", "Drama", "s2"))
    g.add(Triple("Inception", "DIRECTED_BY", "Christopher Nolan", "s3"))
    return g


def test_attribute_objects_are_not_seeds():
    # "Drama" is the object of HAS_GENRE -> excluded from seeds; the film "Drama Queen" still seeds
    r = GraphRetriever(_attr_store(), attribute_relations=frozenset({"HAS_GENRE"}))
    assert r.seeds("a drama queen film") == ["Drama Queen"]
    assert "Drama" not in r.seeds("drama")  # bare attribute value never seeds


def test_seeds_capped_and_ranked_by_specificity():
    g = InMemoryGraphStore()
    g.add(Triple("Batman", "HAS_THEME", "x", "s"))
    g.add(Triple("Batman Begins", "HAS_THEME", "y", "s"))
    # both match "the batman begins story"; the more specific (more tokens) ranks first
    r = GraphRetriever(g, top_k_seeds=1)
    assert r.seeds("the batman begins story") == ["Batman Begins"]


def test_seeds_default_matches_legacy_behavior():
    # with no attribute_relations, seeds == the old etok<=qtok match set
    g = _store()
    assert GraphRetriever(g).seeds("themes in Inception") == ["Inception"]
    assert GraphRetriever(g).seeds("how many films are there") == []
