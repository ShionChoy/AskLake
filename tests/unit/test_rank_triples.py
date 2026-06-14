from engine.graph.retriever import rank_triples
from engine.ports.graph_store import Triple


def test_rank_prefers_target_relation_and_penalizes_hubs():
    t_target = Triple("Film", "ACTED_IN", "Actor", "s1")
    t_hub = Triple("Film", "HAS_GENRE", "Drama", "s2")
    degree_of = {"Film": 5, "Actor": 3, "Drama": 9999}
    ranked = rank_triples([t_hub, t_target], {"Film"}, {"ACTED_IN"}, degree_of)
    assert ranked[0] == t_target


def test_rank_dedups_preserving_best_order():
    t = Triple("A", "R", "B", "s")
    ranked = rank_triples([t, t], {"A"}, frozenset(), {"A": 1, "B": 1})
    assert ranked == [t]
