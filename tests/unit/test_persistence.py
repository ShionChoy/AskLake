from engine.graph.persistence import append_triples, load_store, save_triples
from engine.ports.graph_store import Triple


def test_append_triples_adds_without_clobbering(tmp_path):
    out = tmp_path / "triples.jsonl"
    save_triples([Triple("A", "REL", "B", "s1")], out)
    append_triples([Triple("C", "REL", "D", "s2")], out)
    rows = {(t.subject, t.obj) for t in load_store(out).triples()}
    assert rows == {("A", "B"), ("C", "D")}
