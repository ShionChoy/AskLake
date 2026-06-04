"""Offline, deterministic eval demonstration for CI and `make eval`.

FakeLLMProvider with canned per-case outputs so baseline-vs-agentic runs with NO API key. The
baseline gets `top_rated` wrong (a non-existent `rating` column); the agentic system gets the same
wrong SQL first, then self-corrects -> agentic execution-accuracy > baseline, deterministically.
The REAL benchmark (real LLM over a BIRD/Spider subset) is documented in docs/eval.md; reuse
eval.systems.run_baseline / run_agentic with a real LLMProvider + real EvalCases to reproduce it."""

from __future__ import annotations

from engine.llm.fake import FakeLLMProvider
from eval.cases import MINI_CASES
from eval.harness import SystemReport, evaluate
from eval.systems import run_agentic, run_baseline

_BASELINE = {
    "top_rated": ["SELECT title, rating FROM movies ORDER BY rating DESC LIMIT 1"],  # wrong column
    "count_by_year": [
        "SELECT startYear, COUNT(*) AS n FROM movies GROUP BY startYear ORDER BY startYear"
    ],
    "post_2010": ["SELECT title FROM movies WHERE startYear > 2010 ORDER BY title"],
}

_AGENTIC = {
    "top_rated": [
        "SELECT title, rating FROM movies ORDER BY rating DESC LIMIT 1",  # wrong, then corrected:
        "SELECT title, averageRating FROM movies ORDER BY averageRating DESC LIMIT 1",
    ],
    "count_by_year": [
        "SELECT startYear, COUNT(*) AS n FROM movies GROUP BY startYear ORDER BY startYear"
    ],
    "post_2010": ["SELECT title FROM movies WHERE startYear > 2010 ORDER BY title"],
}


def run_hermetic_comparison() -> tuple[SystemReport, SystemReport]:
    def baseline_one(case, backend):
        return run_baseline(FakeLLMProvider(responses=_BASELINE[case.name]), backend, case.question)

    def agentic_one(case, backend):
        return run_agentic(
            FakeLLMProvider(responses=_AGENTIC[case.name]), backend, case.question, max_retries=2
        )

    baseline = evaluate("baseline (single-prompt)", MINI_CASES, baseline_one)
    agentic = evaluate("agentic (self-correct)", MINI_CASES, agentic_one)
    return baseline, agentic
