from engine.graph.intent import IntentResolver
from engine.graph.neo4j_retriever import Neo4jGraphRetriever
from engine.graph.ontology import load_ontology
from engine.ports.graph_store import Triple

_ATTR = frozenset({"HAS_GENRE", "RELEASED_IN"})
_CONN = frozenset({"HAS_THEME"})
_ROLES = {"HAS_GENRE": {"object": "Genre"}, "RELEASED_IN": {"object": "Year"}}


def _resolver():
    return IntentResolver(load_ontology("datasets/imdb/graph/ontology.yaml"))


class FakeStore:
    def __init__(self, *, seedable=(), neighbors=None, shared=None, degrees=None):
        self._seedable = list(seedable)
        self._n = neighbors or {}
        self._shared = shared or []
        self._deg = degrees or {}

    def seedable_names(self, attr_types):
        return self._seedable

    def neighbors(self, e, *, limit=None, relations=None):
        ts = self._n.get(e, [])
        if relations is not None:
            rels = set(relations)
            ts = [t for t in ts if t.relation in rels]
        return tuple(ts[:limit] if limit else ts)

    def degree(self, e):
        return self._deg.get(e, len(self._n.get(e, [])))

    def degrees(self, names):
        return {n: self.degree(n) for n in names}

    def shared(self, seeds, relations):
        return list(self._shared)


def _retriever(store):
    return Neo4jGraphRetriever(
        store,
        _resolver(),
        attribute_relations=_ATTR,
        connective_relations=_CONN,
        relation_roles=_ROLES,
    )


def test_no_seed_returns_empty():
    assert _retriever(FakeStore(seedable=["Oppenheimer"])).retrieve("hello world").triples == ()


def test_entity_lookup_returns_only_target_relations():
    actor = Triple("Cillian Murphy", "ACTED_IN", "Oppenheimer", "imdb:tt1")
    genre = Triple("Oppenheimer", "HAS_GENRE", "Drama", "imdb:tt1")
    store = FakeStore(seedable=["Oppenheimer"], neighbors={"Oppenheimer": [actor, genre]})
    sg = _retriever(store).retrieve("who acted in Oppenheimer")
    rels = {t.relation for t in sg.triples}
    assert "ACTED_IN" in rels and "HAS_GENRE" not in rels


def test_pairwise_uses_store_shared():
    shared = [
        Triple("Inception", "HAS_THEME", "time", "wiki:1"),
        Triple("Interstellar", "HAS_THEME", "time", "wiki:2"),
    ]
    store = FakeStore(
        seedable=["Inception", "Interstellar"],
        shared=shared,
        neighbors={"Inception": [], "Interstellar": []},
    )
    sg = _retriever(store).retrieve("what themes do Inception and Interstellar share")
    assert set(sg.triples) == set(shared)


def test_seeds_delegates_to_linker():
    store = FakeStore(seedable=["Inception"])
    assert _retriever(store).seeds("inception") == ["Inception"]
