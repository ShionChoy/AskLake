from __future__ import annotations

from eval.harness import EvalCase, SystemReport


def test_evalcase_tier_defaults_empty():
    c = EvalCase(name="x", schema_sql="", question="q", gold_sql="SELECT 1")
    assert c.tier == ""
    c2 = EvalCase(name="y", schema_sql="", question="q", gold_sql="SELECT 1", tier="topn")
    assert c2.tier == "topn"


def test_systemreport_per_tier_defaults_none():
    r = SystemReport(
        name="baseline", n=3, valid_sql_rate=1.0, execution_accuracy=0.5, avg_attempts=0.0
    )
    assert r.per_tier is None
    r2 = SystemReport(
        name="agentic",
        n=3,
        valid_sql_rate=1.0,
        execution_accuracy=0.5,
        avg_attempts=0.0,
        per_tier={"topn": 0.5},
    )
    assert r2.per_tier == {"topn": 0.5}


def test_system_report_has_optional_cost_fields():
    from eval.harness import SystemReport

    r = SystemReport(name="x", n=1, valid_sql_rate=1.0, execution_accuracy=1.0, avg_attempts=0.0)
    assert r.avg_llm_calls == 0.0 and r.avg_wall_ms == 0.0  # additive defaults
    r2 = SystemReport(
        name="y",
        n=1,
        valid_sql_rate=1.0,
        execution_accuracy=1.0,
        avg_attempts=0.0,
        avg_llm_calls=3.0,
        avg_wall_ms=12.5,
    )
    assert r2.avg_llm_calls == 3.0 and r2.avg_wall_ms == 12.5


def test_counting_llm_counts_and_delegates():
    from engine.llm.fake import FakeLLMProvider
    from eval._counting import CountingLLM

    inner = FakeLLMProvider(responses=["SELECT 1", "SELECT 2"])
    c = CountingLLM(inner)
    assert c.complete("a") == "SELECT 1"
    assert c.complete("b", system="s") == "SELECT 2"
    assert c.calls == 2
