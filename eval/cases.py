"""Tiny self-contained eval set (DuckDB schema + NL question + gold SQL).

Hermetic and dataset-agnostic: proves the eval methodology in CI. Swap in a real BIRD/Spider
subset (same EvalCase shape) for the headline numbers; see docs/eval.md."""

from __future__ import annotations

from eval.harness import EvalCase

_SCHEMA = """
CREATE TABLE movies AS SELECT * FROM (VALUES
    ('Alpha', 8.9, 2014),
    ('Beta', 7.5, 2012),
    ('Gamma', 6.0, 2001)
) t(title, averageRating, startYear);
"""

MINI_CASES: list[EvalCase] = [
    EvalCase(
        name="top_rated",
        schema_sql=_SCHEMA,
        question="What is the highest-rated movie title and its rating?",
        gold_sql="SELECT title, averageRating FROM movies ORDER BY averageRating DESC LIMIT 1",
    ),
    EvalCase(
        name="count_by_year",
        schema_sql=_SCHEMA,
        question="How many movies are there per start year?",
        gold_sql=(
            "SELECT startYear, COUNT(*) AS n FROM movies GROUP BY startYear ORDER BY startYear"
        ),
    ),
    EvalCase(
        name="post_2010",
        schema_sql=_SCHEMA,
        question="List movie titles released after 2010, alphabetically.",
        gold_sql="SELECT title FROM movies WHERE startYear > 2010 ORDER BY title",
    ),
]
