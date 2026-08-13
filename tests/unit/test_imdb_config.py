from pathlib import Path

from engine.governance.policy import load_policy
from engine.semantic.semantic_model import load_semantic_layer

_ROOT = Path(__file__).resolve().parents[2]
_SEMANTIC = _ROOT / "datasets" / "imdb" / "semantic.yaml"
_GOV = _ROOT / "datasets" / "imdb" / "governance.yaml"


def test_imdb_semantic_layer_parses():
    layer = load_semantic_layer(_SEMANTIC)
    names = {t.name for t in layer.tables}
    assert {"title_basics", "title_ratings", "name_basics"} <= names
    assert layer.synonyms.get("score") == "averageRating"
    assert layer.few_shots  # at least one example


def test_imdb_governance_policy_parses():
    policy = load_policy(_GOV)
    assert "birthYear" in policy.pii_columns
    assert "public" in policy.mask_roles
    assert policy.require_limit is True
    assert policy.row_filters["public"][0].column == "titleType"


def test_imdb_semantic_has_link_annotations():
    from engine.semantic.semantic_model import load_semantic_layer

    layer = load_semantic_layer("datasets/imdb/semantic.yaml")
    links = {c.name: c.link for t in layer.tables for c in t.columns if c.link}
    assert links.get("genres") == "categorical"
    assert links.get("category") == "categorical"
    assert links.get("primaryName") == "entity"


def test_imdb_semantic_documents_title_type():
    layer = load_semantic_layer(_SEMANTIC)
    cols = {c.name for t in layer.tables for c in t.columns}
    assert "titleType" in cols
    assert layer.synonyms.get("tv series") == "tvSeries"
