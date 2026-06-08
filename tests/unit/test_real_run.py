from engine.lakehouse.duckdb_backend import DuckDBBackend
from engine.llm.fake import FakeLLMProvider
from engine.semantic.semantic_model import SemanticLayer
from eval.harness import EvalCase, SystemReport
from eval.real_run import run_real_eval

_SEED = """
CREATE TABLE movies AS SELECT * FROM (VALUES
    ('Alpha', 8.9), ('Beta', 7.5)
) t(title, averageRating);
"""

_CASES = [
    EvalCase(
        name="top",
        schema_sql="",
        question="highest rated movie",
        gold_sql="SELECT title FROM movies ORDER BY averageRating DESC LIMIT 1",
    )
]


def _backend() -> DuckDBBackend:
    b = DuckDBBackend()
    b.setup(_SEED)
    return b


def test_run_real_eval_returns_three_reports():
    # FakeLLM emits the correct SQL for every system -> all three score 100% exec-acc.
    correct = "SELECT title FROM movies ORDER BY averageRating DESC LIMIT 1"
    llm = FakeLLMProvider(responses=[correct])
    reports = run_real_eval(llm, _backend(), _CASES, SemanticLayer())
    assert isinstance(reports, list) and len(reports) == 3
    names = [r.name for r in reports]
    assert names == ["baseline", "agentic", "semantic"]
    for r in reports:
        assert isinstance(r, SystemReport)
        assert r.n == 1
        assert r.execution_accuracy == 1.0
        assert r.valid_sql_rate == 1.0


def test_run_real_eval_counts_failing_case_not_crash():
    # An LLM whose every call raises must NOT abort the run: the case is counted as a failure
    # (0% exec-acc) and three reports are still returned.
    class _BoomLLM:
        def complete(self, prompt, system=None):
            raise RuntimeError("simulated LLM/network failure")

    reports = run_real_eval(_BoomLLM(), _backend(), _CASES, SemanticLayer())
    assert [r.name for r in reports] == ["baseline", "agentic", "semantic"]
    for r in reports:
        assert r.n == 1
        assert r.execution_accuracy == 0.0
        assert r.valid_sql_rate == 0.0


def test_apply_duckdb_guardrails_runs_on_real_backend():
    from eval.real_run import apply_duckdb_guardrails

    b = _backend()
    apply_duckdb_guardrails(b)  # must not raise
    # backend still works after limits applied
    assert b.run_sql("SELECT 1").rows == [(1,)]


def test_run_real_eval_reports_per_tier():
    correct = "SELECT title FROM movies ORDER BY averageRating DESC LIMIT 1"
    cases = [
        EvalCase(name="a", schema_sql="", question="q", gold_sql=correct, tier="topn"),
        EvalCase(name="b", schema_sql="", question="q", gold_sql=correct, tier="aggregation"),
    ]
    llm = FakeLLMProvider(responses=[correct])
    reports = run_real_eval(llm, _backend(), cases, SemanticLayer())
    for r in reports:
        assert r.per_tier == {"topn": 1.0, "aggregation": 1.0}


def test_run_real_eval_per_tier_none_when_untiered():
    # The existing _CASES have tier="" -> no per-tier breakdown.
    correct = "SELECT title FROM movies ORDER BY averageRating DESC LIMIT 1"
    llm = FakeLLMProvider(responses=[correct])
    reports = run_real_eval(llm, _backend(), _CASES, SemanticLayer())
    for r in reports:
        assert r.per_tier is None
