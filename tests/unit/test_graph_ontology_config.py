from pathlib import Path

from engine.graph.ontology import load_ontology

_ROOT = Path(__file__).resolve().parents[2]
_ONT = _ROOT / "datasets" / "imdb" / "graph" / "ontology.yaml"


def test_imdb_ontology_parses():
    ont = load_ontology(_ONT)
    assert "HAS_THEME" in ont.relation_types
    assert "DIRECTED_BY" in ont.relation_types
    assert ont.entity_types  # at least one entity type
    assert ont.hint  # extraction hint present


def test_imdb_ontology_drops_language_and_country():
    from engine.graph.ontology import load_ontology

    o = load_ontology("datasets/imdb/graph/ontology.yaml")
    assert "IN_LANGUAGE" not in o.relation_types
    assert "FROM_COUNTRY" not in o.relation_types
    assert "IN_LANGUAGE" not in o.attribute_relations
    assert "FROM_COUNTRY" not in o.attribute_relations
    for rel in (
        "HAS_GENRE",
        "RELEASED_IN",
        "DIRECTED_BY",
        "ACTED_IN",
        "FEATURES_CHARACTER",
        "PLAYED_BY",
        "HAS_THEME",
    ):
        assert rel in o.relation_types


def test_ontology_parses_node_roles_and_intents():
    from engine.graph.ontology import load_ontology

    o = load_ontology("datasets/imdb/graph/ontology.yaml")
    assert "HAS_THEME" in o.connective_relations
    assert "HAS_THEME" not in o.attribute_relations  # moved to connective in PR2
    assert "ACTED_IN" in o.entity_relations and "DIRECTED_BY" in o.entity_relations
    names = {i.name: i for i in o.intents}
    assert {"cast", "director", "themes", "connection"} <= set(names)
    cast = names["cast"]
    assert cast.shape == "entity_lookup"
    assert "ACTED_IN" in cast.target_relations
    assert "actor" in cast.triggers
    assert names["themes"].shape == "cluster"
    assert names["connection"].shape == "pairwise"
