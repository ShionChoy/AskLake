from engine.graph.ontology import GraphOntology, load_ontology


def test_attribute_relations_default_empty():
    assert GraphOntology().attribute_relations == ()


def test_load_ontology_reads_attribute_relations(tmp_path):
    p = tmp_path / "ontology.yaml"
    p.write_text(
        "entity_types: [Film, Genre]\n"
        "relation_types: [HAS_GENRE, DIRECTED_BY]\n"
        "attribute_relations: [HAS_GENRE]\n"
        "hint: hi\n"
    )
    ont = load_ontology(p)
    assert ont.attribute_relations == ("HAS_GENRE",)
    assert ont.relation_types == ("HAS_GENRE", "DIRECTED_BY")


def test_imdb_ontology_declares_attribute_relations():
    ont = load_ontology("datasets/imdb_cmu/graph/ontology.yaml")
    assert set(ont.attribute_relations) == {
        "HAS_GENRE",
        "RELEASED_IN",
    }  # HAS_THEME moved to connective_relations


def test_empty_graph_hint_default_empty():
    assert GraphOntology().empty_graph_hint == ""


def test_load_ontology_reads_empty_graph_hint(tmp_path):
    p = tmp_path / "ontology.yaml"
    p.write_text("relation_types: [X]\nempty_graph_hint: not in the graph; try SQL\n")
    assert load_ontology(p).empty_graph_hint == "not in the graph; try SQL"


def test_imdb_ontology_has_empty_graph_hint():
    ont = load_ontology("datasets/imdb_cmu/graph/ontology.yaml")
    assert ont.empty_graph_hint
    assert "knowledge graph" in ont.empty_graph_hint.lower()
