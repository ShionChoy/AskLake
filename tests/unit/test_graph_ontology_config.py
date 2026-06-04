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
