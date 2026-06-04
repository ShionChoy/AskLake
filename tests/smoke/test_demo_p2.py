from demos.demo_p2 import run_demo_p2


def test_demo_p2_self_corrects_and_beats_baseline():
    out = run_demo_p2()
    assert out["rows"][0][0] == "Alpha"  # corrected query returns ranked rows
    assert "self-correction" in out["narrative"]
    assert out["chart_spec"]["type"] == "bar"
    assert out["agentic_execution_accuracy"] > out["baseline_execution_accuracy"]
