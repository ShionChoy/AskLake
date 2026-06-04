from engine.ports.schema import SchemaProvider
from engine.semantic.retriever import LexicalSchemaRetriever
from engine.semantic.semantic_layer import SemanticLayerProvider
from engine.semantic.semantic_model import (
    ColumnDef,
    FewShot,
    MetricDef,
    SemanticLayer,
    TableDef,
)


def _layer() -> SemanticLayer:
    return SemanticLayer(
        tables=(
            TableDef(
                "ratings", "User ratings.", (ColumnDef("averageRating", "score 1-10", "DOUBLE"),)
            ),
            TableDef("people", "Actors and directors.", (ColumnDef("birthYear", "year of birth"),)),
        ),
        metrics=(MetricDef("top", "ORDER BY averageRating DESC", "rank by rating"),),
        synonyms={"score": "averageRating"},
        few_shots=(
            FewShot(
                "best by score",
                "SELECT averageRating FROM ratings ORDER BY averageRating DESC LIMIT 1",
            ),
            FewShot("oldest person", "SELECT birthYear FROM people ORDER BY birthYear LIMIT 1"),
        ),
    )


def test_retriever_prefers_question_relevant_table():
    ctx = LexicalSchemaRetriever().select("films by score", _layer())
    assert ctx.tables[0].name == "ratings"
    # synonym expansion (score -> averageRating) drives the match
    assert any("averageRating" in c.name for c in ctx.tables[0].columns)


def test_retriever_picks_relevant_few_shot():
    ctx = LexicalSchemaRetriever().select("best by score", _layer())
    assert ctx.few_shots[0].question == "best by score"


def test_provider_is_schema_provider_and_grounds():
    provider = SemanticLayerProvider(_layer())
    assert isinstance(provider, SchemaProvider)
    ctx = provider.schema_context("best movie by score")
    assert "averageRating" in ctx
    assert "score -> averageRating" in ctx  # synonyms surfaced
    assert "Metrics:" in ctx  # metric definitions surfaced
    assert "Examples:" in ctx and "SELECT averageRating" in ctx  # few-shot surfaced


def test_provider_handles_empty_layer():
    # never raises; degrades to a minimal context
    assert "Tables:" in SemanticLayerProvider(SemanticLayer()).schema_context("anything")
