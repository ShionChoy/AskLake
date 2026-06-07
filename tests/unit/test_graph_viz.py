import json

from ui import graph_viz
from ui.graph_viz import COLOR_LEAF, COLOR_SUBJECT, MAX_TRIPLES, build_network_data

TRIPLES = [
    ["The Dark Knight", "HAS_THEME", "chaos", "cmu:1"],
    ["The Dark Knight", "HAS_THEME", "identity", "cmu:1"],
    ["Inception", "HAS_THEME", "identity", "cmu:2"],
]


def test_dedupes_nodes_and_counts_degree():
    nodes, edges, truncated = build_network_data(TRIPLES)
    assert truncated is False
    ids = [n["id"] for n in nodes]
    assert ids == ["The Dark Knight", "chaos", "identity", "Inception"]  # first-seen, deduped
    by_id = {n["id"]: n for n in nodes}
    assert by_id["The Dark Knight"]["title"] == "The Dark Knight · 2 connection(s)"
    assert by_id["identity"]["title"] == "identity · 2 connection(s)"
    # a hub (degree 2) renders larger than a leaf (degree 1)
    assert by_id["The Dark Knight"]["size"] > by_id["chaos"]["size"]
    assert len(edges) == 3


def test_two_tone_color():
    nodes, _, _ = build_network_data(TRIPLES)
    color = {n["id"]: n["color"] for n in nodes}
    assert color["The Dark Knight"] == COLOR_SUBJECT
    assert color["Inception"] == COLOR_SUBJECT
    assert color["chaos"] == COLOR_LEAF
    assert color["identity"] == COLOR_LEAF


def test_edge_label_and_source():
    _, edges, _ = build_network_data(TRIPLES)
    e = edges[0]
    assert e["from"] == "The Dark Knight"
    assert e["to"] == "chaos"
    assert e["label"] == "HAS_THEME"
    assert e["title"] == "cmu:1"


def test_truncates_over_cap():
    big = [[f"s{i}", "REL", f"o{i}", f"cmu:{i}"] for i in range(MAX_TRIPLES + 50)]
    _, edges, truncated = build_network_data(big)
    assert truncated is True
    assert len(edges) == MAX_TRIPLES  # only the first MAX_TRIPLES are used


def test_skips_malformed_rows():
    rows = [
        ["A", "REL", "B", "cmu:1"],
        ["", "REL", "B", "cmu:2"],  # missing subject
        ["A", "REL", "", "cmu:3"],  # missing object
        ["A", "REL"],  # too short
    ]
    nodes, edges, _ = build_network_data(rows)
    assert len(edges) == 1
    assert {n["id"] for n in nodes} == {"A", "B"}


def test_self_loop_counts_once():
    nodes, edges, _ = build_network_data([["A", "IS_A", "A", "src:1"]])
    by_id = {n["id"]: n for n in nodes}
    assert by_id["A"]["title"] == "A · 1 connection(s)"
    assert len(edges) == 1


def test_mixed_node_gets_subject_color():
    # "bridge" is first an object (A->bridge), then a subject (bridge->C)
    nodes, _, _ = build_network_data([["A", "R", "bridge", "s"], ["bridge", "R", "C", "s"]])
    color = {n["id"]: n["color"] for n in nodes}
    assert color["bridge"] == COLOR_SUBJECT


def test_to_html_contains_labels():
    nodes, edges, _ = build_network_data(TRIPLES)
    html = graph_viz._to_html(nodes, edges)
    assert isinstance(html, str) and len(html) > 0
    assert "The Dark Knight" in html  # node label embedded in the vis.js data
    assert "HAS_THEME" in html  # edge label embedded


def test_node_scale_scales_size():
    base, _, _ = build_network_data(TRIPLES)
    scaled, _, _ = build_network_data(TRIPLES, node_scale=2.0)
    base_size = {n["id"]: n["size"] for n in base}
    scaled_size = {n["id"]: n["size"] for n in scaled}
    assert scaled_size["The Dark Knight"] == base_size["The Dark Knight"] * 2
    assert scaled_size["chaos"] == base_size["chaos"] * 2


def test_options_spacing_mapping():
    opts = json.loads(graph_viz._options(160))
    bh = opts["physics"]["barnesHut"]
    assert bh["springLength"] == 160
    assert bh["gravitationalConstant"] == -12000
    opts2 = json.loads(graph_viz._options(320))
    bh2 = opts2["physics"]["barnesHut"]
    assert bh2["springLength"] == 320
    assert bh2["gravitationalConstant"] == -24000


def test_freeze_injects_physics_off():
    nodes, edges, _ = build_network_data(TRIPLES)
    frozen = graph_viz._to_html(nodes, edges, freeze=True)
    live = graph_viz._to_html(nodes, edges, freeze=False)
    # The freeze snippet sets physics: false via network.once; this exact call is unique to our
    # injection (the minified vis.js bundle only has "stabilizationIterationsDone" bare).
    assert 'network.once("stabilizationIterationsDone"' in frozen
    assert 'network.once("stabilizationIterationsDone"' not in live
