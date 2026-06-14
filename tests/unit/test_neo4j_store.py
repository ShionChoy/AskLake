import pytest

from engine.graph.neo4j_store import Neo4jGraphStore
from engine.ports.graph_store import Triple


class FakeNeo4j:
    """Driver + session + result triple. `responder(query, params) -> list[dict]`."""

    def __init__(self, responder=None):
        self.calls = []
        self.closed = False
        self._responder = responder or (lambda q, p: [])

    def session(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def run(self, query, **params):
        self.calls.append((query, params))
        return self._responder(query, params)

    def close(self):
        self.closed = True


def test_ensure_schema_creates_constraint_and_fulltext():
    drv = FakeNeo4j()
    Neo4jGraphStore(drv).ensure_schema()
    joined = " ".join(q for q, _ in drv.calls)
    assert "CONSTRAINT entity_name" in joined
    assert "FULLTEXT INDEX entity_fulltext" in joined


def test_add_interpolates_typed_labels_and_relation():
    drv = FakeNeo4j()
    store = Neo4jGraphStore(
        drv, relation_roles={"DIRECTED_BY": {"subject": "Film", "object": "Person"}}
    )
    store.add(Triple("Inception", "DIRECTED_BY", "Christopher Nolan", "imdb:tt1"))
    q, p = drv.calls[0]
    assert ":`Film`" in q and ":`Person`" in q and "[r:`DIRECTED_BY`]" in q
    assert p == {"s": "Inception", "o": "Christopher Nolan", "src": "imdb:tt1"}


def test_unsafe_identifier_rejected():
    store = Neo4jGraphStore(FakeNeo4j())
    with pytest.raises(ValueError):
        store.add(Triple("a", "BAD-REL", "b", "s"))


def test_load_triples_batches_grouped_by_relation():
    drv = FakeNeo4j()
    store = Neo4jGraphStore(
        drv,
        relation_roles={
            "HAS_THEME": {"subject": "Film", "object": "Theme"},
            "DIRECTED_BY": {"subject": "Film", "object": "Person"},
        },
    )
    n = store.load_triples(
        [
            Triple("A", "HAS_THEME", "x", "s"),
            Triple("B", "HAS_THEME", "y", "s"),
            Triple("A", "DIRECTED_BY", "Dir", "s"),
        ]
    )
    assert n == 3
    unwinds = [(q, p) for q, p in drv.calls if "UNWIND" in q]
    assert len(unwinds) == 2  # one per relation group
    theme_call = next(p for q, p in unwinds if "HAS_THEME" in q)
    assert theme_call["rows"] == [
        {"s": "A", "o": "x", "src": "s"},
        {"s": "B", "o": "y", "src": "s"},
    ]


def test_neighbors_builds_filter_and_limit():
    captured = {}

    def responder(q, p):
        captured["q"], captured["p"] = q, p
        return [{"s": "Actor", "rel": "ACTED_IN", "o": "Film", "src": "s1"}]

    store = Neo4jGraphStore(FakeNeo4j(responder))
    out = store.neighbors("Film", limit=10, relations=["ACTED_IN"])
    assert "type(r) IN $relations" in captured["q"] and "LIMIT $limit" in captured["q"]
    assert captured["p"]["relations"] == ["ACTED_IN"] and captured["p"]["limit"] == 10
    assert out == (Triple("Actor", "ACTED_IN", "Film", "s1"),)


def test_degree_and_degrees():
    store = Neo4jGraphStore(FakeNeo4j(lambda q, p: [{"deg": 7}]))
    assert store.degree("Film") == 7
    multi = Neo4jGraphStore(
        FakeNeo4j(lambda q, p: [{"name": "A", "deg": 2}, {"name": "B", "deg": 3}])
    )
    assert multi.degrees(["A", "B"]) == {"A": 2, "B": 3}


def test_seedable_names_passes_attr_types():
    captured = {}

    def responder(q, p):
        captured["p"] = p
        return [{"name": "Inception"}, {"name": "Nolan"}]

    store = Neo4jGraphStore(FakeNeo4j(responder))
    assert store.seedable_names(["Genre", "Year"]) == ["Inception", "Nolan"]
    assert captured["p"]["attr"] == ["Genre", "Year"]


def test_shared_returns_triples():
    rows = [{"s": "Inception", "rel": "HAS_THEME", "o": "time", "src": "w1"}]
    store = Neo4jGraphStore(FakeNeo4j(lambda q, p: rows))
    out = store.shared(["Inception", "Interstellar"], ["HAS_THEME"])
    assert out == [Triple("Inception", "HAS_THEME", "time", "w1")]


def test_close_closes_driver():
    drv = FakeNeo4j()
    Neo4jGraphStore(drv).close()
    assert drv.closed is True
