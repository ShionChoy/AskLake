from demos.demo_p4 import run_demo_p4


def test_demo_p4_routes_and_fuses_with_citations():
    out = run_demo_p4()
    # the Router classifies each question correctly
    assert out["routes"]["sql"] == "sql"
    assert out["routes"]["graph"] == "graph"
    assert out["routes"]["fusion"] == "sql+graph"
    # the fused answer carries the SQL film table ...
    assert out["fused"]["path"] == "sql+graph"
    assert out["fused"]["sql_rows"]  # non-empty structured result from the SQL path
    # ... and the graph contribution with a traceable plot citation + a theme
    narrative = out["fused"]["narrative"]
    assert "[graph]" in narrative and "[sql]" in narrative
    assert "[plot_inception]" in narrative
    assert "identity" in narrative  # a common theme surfaced via multi-hop from the director
