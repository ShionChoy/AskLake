from demos.demo_p5 import run_demo_p5


def test_demo_p5_collects_observability_metrics():
    out = run_demo_p5()
    # the self-correction made exactly two LLM calls (bad column -> corrected)
    assert out["llm_calls"] == 2.0
    # the first candidate's bad column raised one caught SQL error
    assert out["sql_errors"] == 1.0
    # both candidates were executed against the backend
    assert out["storage_runs"] == 2.0
    # the corrected query returned rows
    assert out["rows"]
    # Prometheus exposition is produced and self-describing
    assert "asklake_events_total" in out["exposition"]
