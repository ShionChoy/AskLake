from demos.demo_p0 import run_demo_p0


def test_demo_p0_runs_and_returns_expected_top_row():
    result = run_demo_p0()
    assert result["columns"] == ["title", "rating"]
    assert result["rows"][0][0] == "Interstellar"
    assert len(result["rows"]) == 2
