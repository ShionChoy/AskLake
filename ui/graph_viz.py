from __future__ import annotations

import json

MAX_TRIPLES = 200
# vis.js node color dicts (Morandi muted palette): cool dusty blue-gray for entities,
# warm dusty clay for leaf concepts. background + darker border + hover/highlight variants.
COLOR_SUBJECT = {  # entities with outgoing edges (e.g. films/titles)
    "background": "#88A0A8",
    "border": "#6E848B",
    "highlight": {"background": "#9CB2B9", "border": "#6E848B"},
    "hover": {"background": "#9CB2B9", "border": "#6E848B"},
}
COLOR_LEAF = {  # leaf concepts that only appear as objects (e.g. themes)
    "background": "#C5A99B",
    "border": "#A88B7D",
    "highlight": {"background": "#D4BCB1", "border": "#A88B7D"},
    "hover": {"background": "#D4BCB1", "border": "#A88B7D"},
}
_SIZE_BASE = 10
_SIZE_PER_DEGREE = 4
_SIZE_MAX = 40


def build_network_data(
    triples: list[list], *, node_scale: float = 1.0
) -> tuple[list[dict], list[dict], bool]:
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
            "size": round(
                min(_SIZE_BASE + _SIZE_PER_DEGREE * degree[name], _SIZE_MAX) * node_scale
            ),
            "color": COLOR_SUBJECT if name in subjects else COLOR_LEAF,
        }
        for name in order
    ]
    return nodes, edges, truncated


# vis.js options (strict JSON via json.dumps), parameterized by layout spacing.
# springLength = spacing; gravitationalConstant = -75 * spacing (so spacing=160 reproduces the
# -12000 default). hover + hoverConnectedEdges highlights a node's neighborhood with no custom JS.
def _options(spacing: int = 160) -> str:
    opts = {
        "interaction": {
            "hover": True,
            "hoverConnectedEdges": True,
            "tooltipDelay": 120,
            "navigationButtons": False,
            "zoomView": True,
            "dragNodes": True,
        },
        "physics": {
            "solver": "barnesHut",
            "barnesHut": {
                "gravitationalConstant": -75 * spacing,
                "centralGravity": 0.25,
                "springLength": spacing,
                "springConstant": 0.05,
                "damping": 0.09,
                "avoidOverlap": 0.6,
            },
            "stabilization": {"iterations": 200},
        },
        "nodes": {
            "shape": "dot",
            "borderWidth": 2,
            "borderWidthSelected": 3,
            "font": {
                "size": 14,
                "color": "#2A2A2A",
                "strokeWidth": 4,
                "strokeColor": "#ffffff",
                "face": "Helvetica",
            },
            "shadow": {"enabled": True, "size": 6, "x": 0, "y": 2, "color": "rgba(0,0,0,0.15)"},
        },
        "edges": {
            "color": {"color": "#C2CAD6", "highlight": "#6B7B8F", "hover": "#6B7B8F"},
            "width": 1.5,
            "selectionWidth": 2,
            "smooth": {"type": "continuous", "roundness": 0.2},
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.5}},
            "font": {
                "size": 12,
                "color": "#3E4654",
                "strokeWidth": 5,
                "strokeColor": "#ffffff",
                "align": "horizontal",
            },
        },
    }
    return json.dumps(opts)


# Lets vis.js lay the graph out, then turns physics off so nodes stop jiggling (freeze).
_FREEZE_JS = (
    'network.once("stabilizationIterationsDone", '
    "function () { network.setOptions({ physics: false }); });"
)


def _to_html(
    nodes: list[dict], edges: list[dict], *, spacing: int = 160, freeze: bool = False
) -> str:
    """Build a self-contained vis.js HTML document from pre-computed nodes/edges.
    cdn_resources='in_line' inlines the JS so the canvas renders offline. `spacing` tunes the
    layout; `freeze=True` stops the physics jiggle once the graph has laid itself out."""
    from pyvis.network import Network

    net = Network(
        height="520px",
        width="100%",
        directed=True,
        bgcolor="#FAFBFC",
        font_color="#222222",
        notebook=False,
        cdn_resources="in_line",
    )
    net.set_options(_options(spacing))
    for n in nodes:
        net.add_node(n["id"], label=n["label"], title=n["title"], size=n["size"], color=n["color"])
    for e in edges:
        net.add_edge(e["from"], e["to"], label=e["label"], title=e["title"])
    html = net.generate_html(notebook=False)
    if freeze and "return network;" in html:
        html = html.replace("return network;", _FREEZE_JS + "\n  return network;", 1)
    return html


def render_network(triples: list[list]) -> None:
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
    # +20px over the 520px canvas leaves an iframe buffer so the canvas isn't clipped
    components.html(html, height=540, scrolling=False)
