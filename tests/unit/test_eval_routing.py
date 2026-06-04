from eval.routing_hermetic import run_graph_grounding, run_routing_eval


def test_router_picks_the_expected_path_on_every_case():
    report = run_routing_eval()
    assert report.n == 3
    assert report.accuracy == 1.0
    actual = {row[0]: row[2] for row in report.rows}  # question -> actual
    assert set(actual.values()) == {"sql", "graph", "sql+graph"}


def test_graph_grounding_lift_is_illustrative_zero_to_one():
    no_graph, grounded = run_graph_grounding()
    assert no_graph == 0  # SQL-only has no plot-theme data to cite
    assert grounded == 1  # the graph path returns a cited theme
