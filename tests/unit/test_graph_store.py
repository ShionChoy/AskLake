from engine.graph.store import InMemoryGraphStore
from engine.ports.graph_store import GraphStore, Triple


def _store() -> InMemoryGraphStore:
    g = InMemoryGraphStore()
    g.add(Triple("Inception", "HAS_THEME", "dreams", "plot_inception"))
    g.add(Triple("Inception", "DIRECTED_BY", "Christopher Nolan", "plot_inception"))
    g.add(Triple("Interstellar", "HAS_THEME", "time", "plot_interstellar"))
    return g


def test_is_graph_store():
    assert isinstance(_store(), GraphStore)


def test_triples_roundtrip():
    g = _store()
    assert len(g.triples()) == 3
    assert Triple("Interstellar", "HAS_THEME", "time", "plot_interstellar") in g.triples()


def test_neighbors_touch_entity_on_either_side():
    g = _store()
    # 'Inception' appears as subject of two triples
    assert len(g.neighbors("Inception")) == 2
    # 'Christopher Nolan' appears as object of one triple
    assert len(g.neighbors("Christopher Nolan")) == 1
    assert g.neighbors("nobody") == ()


def test_entities_union_of_subjects_and_objects():
    g = _store()
    assert g.entities() == frozenset(
        {"Inception", "dreams", "Christopher Nolan", "Interstellar", "time"}
    )
