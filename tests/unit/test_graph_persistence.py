import pytest

from engine.graph.persistence import load_store, save_triples
from engine.ports.graph_store import Triple


def test_save_then_load_roundtrips(tmp_path):
    triples = [
        Triple("The Dark Knight", "DIRECTED_BY", "Christopher Nolan", "wiki:1"),
        Triple("The Dark Knight", "HAS_THEME", "identity", "wiki:1"),
    ]
    p = tmp_path / "g" / "triples.jsonl"  # nested dir must be created
    save_triples(triples, p)
    store = load_store(p)
    assert set(store.triples()) == set(triples)
    assert "The Dark Knight" in store.entities()
    assert "identity" in store.entities()
    # source (citation) is preserved across the round-trip
    assert {t.source for t in store.triples()} == {"wiki:1"}


def test_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_store(tmp_path / "nope.jsonl")
