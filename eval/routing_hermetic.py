"""Hermetic Phase-4 eval. (1) Routing accuracy: does the heuristic Router send each question to
the right path (sql / graph / fused)? (2) Graph-grounding lift: a plot-theme question only the
graph path can answer with a citation. Canned + deterministic, like eval.hermetic /
eval.semantic_hermetic; real numbers come from the manual run (docs/eval.md)."""

from __future__ import annotations

from dataclasses import dataclass

from engine.graph.store import InMemoryGraphStore
from engine.ports.graph_store import Triple
from engine.ports.retrieval import RetrievalResult
from engine.retrieval.graph_rag_path import GraphRagPath
from engine.retrieval.router import Router


@dataclass(frozen=True)
class RoutingReport:
    n: int
    accuracy: float
    rows: tuple[tuple[str, str, str, bool], ...]  # (question, expected, actual, ok)


class _SqlStub:
    """Stand-in SQL path: route() never executes it; only the routing decision is measured."""

    name = "sql"

    def can_handle(self, question: str) -> bool:
        return True

    def run(self, question: str) -> RetrievalResult:  # pragma: no cover - not invoked by route()
        raise NotImplementedError


_ROUTING_CASES = [
    ("How many films did Christopher Nolan direct?", "sql"),
    ("What are the common themes in Inception?", "graph"),
    ("Nolan's highest-rated films before 2013 and their common plot themes", "sql+graph"),
]


def _graph() -> InMemoryGraphStore:
    g = InMemoryGraphStore()
    g.add(Triple("Inception", "HAS_THEME", "dreams", "plot_inception"))
    g.add(Triple("Inception", "DIRECTED_BY", "Christopher Nolan", "plot_inception"))
    g.add(Triple("Interstellar", "HAS_THEME", "time", "plot_interstellar"))
    return g


def run_routing_eval() -> RoutingReport:
    store = _graph()
    router = Router(_SqlStub(), GraphRagPath(store), entity_vocab=store.entities())
    rows: list[tuple[str, str, str, bool]] = []
    correct = 0
    for question, expected in _ROUTING_CASES:
        d = router.route(question)
        actual = "+".join(d.paths) if d.fuse else d.paths[0]
        ok = actual == expected
        correct += int(ok)
        rows.append((question, expected, actual, ok))
    n = len(_ROUTING_CASES)
    return RoutingReport(n=n, accuracy=correct / n if n else 0.0, rows=tuple(rows))


def run_graph_grounding() -> tuple[int, int]:
    """Illustrative grounding lift on a 'themes' question. The no-graph baseline (SQL over the
    structured tables) has no plot-theme data to cite (0); the graph path returns a cited theme
    (1)."""
    store = _graph()
    rr = GraphRagPath(store).run("What are the themes in Inception?")
    grounded = 1 if (rr.result and rr.result.rows) else 0
    no_graph = 0
    return no_graph, grounded
