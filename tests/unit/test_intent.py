from engine.graph.intent import IntentResolver
from engine.graph.ontology import load_ontology


def _resolver():
    return IntentResolver(load_ontology("datasets/imdb_cmu/graph/ontology.yaml"))


def test_cast_question_resolves_to_cast_intent():
    i = _resolver().resolve("who are the actors in Inception")
    assert i.name == "cast" and i.shape == "entity_lookup"


def test_theme_question_resolves_to_cluster():
    assert _resolver().resolve("what are the themes of Inception").shape == "cluster"


def test_director_question_resolves_to_entity_lookup():
    i = _resolver().resolve("who directed The Dark Knight")
    assert i.name == "director" and "DIRECTED_BY" in i.target_relations


def test_connection_question_resolves_to_pairwise():
    assert _resolver().resolve("what do A and B have in common").shape == "pairwise"


def test_no_trigger_falls_back_to_open_over_all_relations():
    i = _resolver().resolve("Inception")
    assert i.shape == "open"
    assert "HAS_THEME" in i.target_relations and "ACTED_IN" in i.target_relations
