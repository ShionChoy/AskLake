from engine.graph.intent import IntentResolver
from engine.graph.ontology import GraphOntology, Intent
from engine.graph.retriever import GraphRetriever
from engine.graph.store import InMemoryGraphStore
from engine.ports.graph_store import Triple


def _movie_store():
    g = InMemoryGraphStore()
    g.add(Triple("Inception", "DIRECTED_BY", "Christopher Nolan", "s"))
    g.add(Triple("Inception", "ACTED_IN", "Leonardo DiCaprio", "s"))
    g.add(Triple("Leonardo DiCaprio", "ACTED_IN", "Inception", "s"))
    g.add(Triple("Inception", "FEATURES_CHARACTER", "Cobb", "s"))
    g.add(Triple("Inception", "HAS_THEME", "dreams", "s"))
    g.add(Triple("Inception", "HAS_GENRE", "Sci-Fi", "s"))
    return g


_ONT = GraphOntology(
    relation_types=("DIRECTED_BY", "ACTED_IN", "FEATURES_CHARACTER", "HAS_THEME", "HAS_GENRE"),
    connective_relations=("HAS_THEME",),
    attribute_relations=("HAS_GENRE",),
    intents=(
        Intent(
            "cast",
            frozenset({"actors"}),
            frozenset({"ACTED_IN", "FEATURES_CHARACTER"}),
            "entity_lookup",
        ),
        Intent("themes", frozenset({"themes"}), frozenset({"HAS_THEME"}), "cluster"),
    ),
)


def _retriever(store):
    return GraphRetriever(
        store,
        attribute_relations=frozenset({"HAS_GENRE"}),
        connective_relations=frozenset({"HAS_THEME"}),
        intent_resolver=IntentResolver(_ONT),
    )


def test_cast_and_theme_questions_return_different_subgraphs():
    r = _retriever(_movie_store())
    cast_rels = {t.relation for t in r.retrieve("actors of Inception").triples}
    theme_rels = {t.relation for t in r.retrieve("themes of Inception").triples}
    assert "ACTED_IN" in cast_rels and "HAS_THEME" not in cast_rels
    assert "HAS_THEME" in theme_rels and "ACTED_IN" not in theme_rels


def test_theme_cluster_connects_films_sharing_a_theme():
    g = _movie_store()
    g.add(Triple("Interstellar", "HAS_THEME", "dreams", "s"))
    sg = _retriever(g).retrieve("themes of Inception")
    nodes = {t.subject for t in sg.triples} | {t.obj for t in sg.triples}
    assert "Interstellar" in nodes  # reached via the shared theme node


def test_attribute_object_never_expands():
    g = _movie_store()
    for i in range(200):
        g.add(Triple(f"Other {i}", "HAS_GENRE", "Sci-Fi", "s"))
    triples = _retriever(g).retrieve("themes of Inception").triples
    assert not any(t.subject.startswith("Other ") for t in triples)


def test_open_shape_without_resolver_matches_legacy_bounded_bfs():
    r = GraphRetriever(_movie_store())
    sg = r.retrieve("Inception")
    assert sg.seeds == ("Inception",) and len(sg.triples) > 0
