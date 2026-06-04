"""Offline semantic-vs-raw comparison: the grounding lift the semantic layer adds on a question
with a business synonym ('score') the bare schema doesn't expose. Hermetic + canned, like
eval.hermetic; the bare-schema agentic loop keeps emitting a non-existent `score` column (nothing
grounds it), while the semantic-grounded loop maps score -> averageRating. Real numbers come from
the manual run (docs/eval.md)."""

from __future__ import annotations

from engine.llm.fake import FakeLLMProvider
from engine.semantic.semantic_model import ColumnDef, FewShot, SemanticLayer, TableDef
from eval.harness import EvalCase, SystemReport, evaluate
from eval.systems import run_agentic, run_semantic

_SCHEMA = """
CREATE TABLE movies AS SELECT * FROM (VALUES
    ('Alpha', 8.9), ('Beta', 7.5), ('Gamma', 6.0)
) t(title, averageRating);
"""

_CASES = [
    EvalCase(
        name="best_by_score",
        schema_sql=_SCHEMA,
        question="Which film has the best score?",
        gold_sql="SELECT title, averageRating FROM movies ORDER BY averageRating DESC LIMIT 1",
    ),
]

_LAYER = SemanticLayer(
    tables=(
        TableDef(
            "movies",
            "One row per film.",
            (
                ColumnDef("title", "Film title."),
                ColumnDef("averageRating", "User rating 1-10.", "DOUBLE"),
            ),
        ),
    ),
    synonyms={"score": "averageRating", "film": "movies"},
    few_shots=(
        FewShot(
            "best film by score",
            "SELECT title, averageRating FROM movies ORDER BY averageRating DESC LIMIT 1",
        ),
    ),
)

# Bare schema: no synonym, so the loop keeps guessing the non-existent `score` column.
# FakeLLMProvider cycles its responses, so one entry is sufficient — it will repeat on retries.
_RAW = {"best_by_score": ["SELECT title, score FROM movies ORDER BY score DESC LIMIT 1"]}
# Semantic-grounded: the layer maps score -> averageRating; succeeds on first try.
_SEMANTIC = {
    "best_by_score": ["SELECT title, averageRating FROM movies ORDER BY averageRating DESC LIMIT 1"]
}


def run_semantic_comparison() -> tuple[SystemReport, SystemReport]:
    def raw_one(case, backend):
        return run_agentic(FakeLLMProvider(responses=_RAW[case.name]), backend, case.question)

    def semantic_one(case, backend):
        return run_semantic(
            FakeLLMProvider(responses=_SEMANTIC[case.name]), backend, case.question, _LAYER
        )

    raw = evaluate("raw schema (P2)", _CASES, raw_one)
    semantic = evaluate("semantic layer (P3)", _CASES, semantic_one)
    return raw, semantic
