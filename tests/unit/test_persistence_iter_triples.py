from engine.graph.persistence import iter_triples, load_store
from engine.ports.graph_store import Triple


def test_iter_triples_streams(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text(
        '{"subject":"A","relation":"R","obj":"B","source":"s1"}\n'
        "\n"  # blank line tolerated
        '{"subject":"C","relation":"R","obj":"D"}\n'  # missing source -> ""
    )
    out = list(iter_triples(str(p)))
    assert out == [Triple("A", "R", "B", "s1"), Triple("C", "R", "D", "")]


def test_load_store_still_works(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text('{"subject":"A","relation":"R","obj":"B","source":"s1"}\n')
    store = load_store(str(p))
    assert store.triples() == (Triple("A", "R", "B", "s1"),)
