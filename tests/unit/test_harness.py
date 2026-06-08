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
