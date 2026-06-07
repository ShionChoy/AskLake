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
