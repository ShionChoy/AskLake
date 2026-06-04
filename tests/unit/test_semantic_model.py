from engine.semantic.semantic_model import load_semantic_layer

YAML = """
tables:
  - name: movies
    description: One row per film.
    columns:
      - {name: title, description: Film title.}
      - {name: averageRating, type: DOUBLE, description: User rating 1-10.}
metrics:
  - name: rating
    expression: title_ratings.averageRating
    description: Use averageRating for rating/score.
synonyms:
  score: averageRating
  film: movies
few_shots:
  - question: best film
    sql: SELECT title FROM movies ORDER BY averageRating DESC LIMIT 1
"""


def test_load_semantic_layer(tmp_path):
    p = tmp_path / "semantic.yaml"
    p.write_text(YAML)
    layer = load_semantic_layer(p)
    assert [t.name for t in layer.tables] == ["movies"]
    assert layer.tables[0].columns[1].name == "averageRating"
    assert layer.tables[0].columns[1].type == "DOUBLE"
    assert layer.synonyms["score"] == "averageRating"
    assert layer.metrics[0].name == "rating"
    assert layer.few_shots[0].sql.startswith("SELECT title")


def test_load_empty_file(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("")
    layer = load_semantic_layer(p)
    assert layer.tables == () and layer.synonyms == {}
