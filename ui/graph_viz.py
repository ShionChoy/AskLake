from __future__ import annotations

MAX_TRIPLES = 200
COLOR_SUBJECT = "#4C78A8"  # entities with outgoing edges (e.g. films/titles)
COLOR_LEAF = "#F58518"  # leaf concepts that only appear as objects (e.g. themes)
_SIZE_BASE = 10
_SIZE_PER_DEGREE = 4
_SIZE_MAX = 40


def build_network_data(triples: list[list]) -> tuple[list[dict], list[dict], bool]:
    """Pure transform: triples [subject, relation, object, source] -> (nodes, edges, truncated).

    nodes/edges are plain pyvis-ready dicts; truncated is True when the input exceeded
    MAX_TRIPLES (only the first MAX_TRIPLES are used). Rows shorter than 3 elements, or
    with an empty subject or object, are skipped. Node coloring is structural — a node
    that is ever a subject (has an outgoing edge) vs. a leaf object — never keyed on
    dataset-specific entity types."""
    truncated = len(triples) > MAX_TRIPLES
    rows = triples[:MAX_TRIPLES]

    degree: dict[str, int] = {}
    subjects: set[str] = set()
    order: list[str] = []
    edges: list[dict] = []

    def _see(name: str) -> None:
        if name not in degree:
            degree[name] = 0
            order.append(name)

    for row in rows:
        if len(row) < 3:
            continue
        subject, relation, obj = row[0], row[1], row[2]
        source = row[3] if len(row) > 3 else ""
        if not subject or not obj:
            continue
        _see(subject)
        _see(obj)
        subjects.add(subject)
        degree[subject] += 1
        if obj != subject:
            degree[obj] += 1
        edges.append({"from": subject, "to": obj, "label": str(relation), "title": str(source)})

    nodes = [
        {
            "id": name,
            "label": name,
            "title": f"{name} · {degree[name]} connection(s)",
            "size": min(_SIZE_BASE + _SIZE_PER_DEGREE * degree[name], _SIZE_MAX),
            "color": COLOR_SUBJECT if name in subjects else COLOR_LEAF,
        }
        for name in order
    ]
    return nodes, edges, truncated


# vis.js options as strict JSON (pyvis.set_options does json.loads on this string).
# hover + hoverConnectedEdges gives the "highlight neighbors" feel with no custom JS.
_PYVIS_OPTIONS = """
{
  "interaction": {"hover": true, "hoverConnectedEdges": true, "tooltipDelay": 100,
                  "navigationButtons": true},
  "physics": {"stabilization": {"iterations": 150}},
  "nodes": {"shape": "dot", "font": {"size": 14}},
  "edges": {"arrows": {"to": {"enabled": true}}, "font": {"size": 10, "align": "middle"},
            "smooth": false}
}
"""


def _to_html(nodes: list[dict], edges: list[dict]) -> str:
    """Build a self-contained vis.js HTML document from pre-computed nodes/edges.
    cdn_resources='in_line' inlines the JS so the canvas renders offline."""
    from pyvis.network import Network

    net = Network(
        height="520px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#222222",
        notebook=False,
        cdn_resources="in_line",
    )
    net.set_options(_PYVIS_OPTIONS)
    for n in nodes:
        net.add_node(n["id"], label=n["label"], title=n["title"], size=n["size"], color=n["color"])
    for e in edges:
        net.add_edge(e["from"], e["to"], label=e["label"], title=e["title"])
    return net.generate_html(notebook=False)


def render_network(triples) -> None:
    """Render the interactive network into the current Streamlit container."""
    import streamlit as st
    import streamlit.components.v1 as components

    nodes, edges, truncated = build_network_data(triples)
    if not nodes:
        st.caption("No graph facts to visualize.")
        return
    if truncated:
        st.caption(f"Showing the first {MAX_TRIPLES} facts.")
    try:
        html = _to_html(nodes, edges)
    except Exception:  # noqa: BLE001
        st.caption("Network view unavailable.")
        return
    components.html(html, height=540, scrolling=False)
