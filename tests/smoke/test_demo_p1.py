from demos.demo_p1 import run_demo_p1


def test_demo_p1_returns_ranked_post_2010_scifi():
    out = run_demo_p1()
    assert out["columns"] == ["title", "rating"]
    assert out["rows"][0][0] == "Sci A"  # 8.9, post-2010 -> ranked first
    titles = [r[0] for r in out["rows"]]
    assert "Old Sci" not in titles  # pre-2010 filtered out
    assert out["chart_spec"]["type"] == "bar"
