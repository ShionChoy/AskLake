from pathlib import Path

from engine.graph.ontology import load_ontology

_ROOT = Path(__file__).resolve().parents[2]
_ONT = _ROOT / "datasets" / "imdb_cmu" / "graph" / "ontology.yaml"


def test_imdb_ontology_parses():
    ont = load_ontology(_ONT)
    assert "HAS_THEME" in ont.relation_types
    assert "DIRECTED_BY" in ont.relation_types
    assert ont.entity_types  # at least one entity type
    assert ont.hint  # extraction hint present


def test_imdb_ontology_drops_language_and_country():
    from engine.graph.ontology import load_ontology

    o = load_ontology("datasets/imdb_cmu/graph/ontology.yaml")
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
