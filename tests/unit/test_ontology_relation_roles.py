from engine.graph.ontology import load_ontology


def test_load_ontology_parses_relation_roles(tmp_path):
    p = tmp_path / "ont.yaml"
    p.write_text(
        "relation_types: [DIRECTED_BY]\n"
        "relation_roles:\n"
        "  DIRECTED_BY: {subject: Film, object: Person}\n"
    )
    ont = load_ontology(str(p))
    assert ont.relation_roles == {"DIRECTED_BY": {"subject": "Film", "object": "Person"}}


def test_relation_roles_defaults_to_empty(tmp_path):
    p = tmp_path / "ont.yaml"
    p.write_text("relation_types: [X]\n")
    assert load_ontology(str(p)).relation_roles == {}


def test_imdb_ontology_has_relation_roles():
    ont = load_ontology("datasets/imdb_cmu/graph/ontology.yaml")
    assert ont.relation_roles["ACTED_IN"] == {"subject": "Person", "object": "Film"}
    assert ont.relation_roles["HAS_THEME"] == {"subject": "Film", "object": "Theme"}
