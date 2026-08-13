from __future__ import annotations

import re
from dataclasses import dataclass

from engine.ports.retrieval import RetrievalPath, RetrievalResult
from engine.retrieval.synthesizer import Synthesizer

_SQL_HINTS = frozenset(
    {
        "how",
        "many",
        "count",
        "number",
        "average",
        "avg",
        "sum",
        "total",
        "top",
        "highest",
        "lowest",
        "rated",
        "rating",
        "ratings",
        "score",
        "scores",
        "year",
        "years",
        "before",
        "after",
        "most",
        "least",
        "list",
        "rank",
        "ranked",
        "per",
        "above",
        "below",
    }
)
_GRAPH_HINTS = frozenset(
    {
        "theme",
        "themes",
        "plot",
        "plots",
        "story",
        "stories",
        "about",
        "common",
        "motif",
        "motifs",
        "character",
        "characters",
        "relationship",
        "relationships",
        "related",
        "connect",
        "connection",
        "connects",
        "actor",
        "actors",
        "cast",
        "starred",
        "starring",
        "acted",
        "director",
        "directors",
        "directed",
        "between",
        "share",
        "shared",
        "similar",
    }
)


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


@dataclass(frozen=True)
class RoutingDecision:
    paths: tuple[str, ...]
    fuse: bool


class Router:
    """Score SQL-vs-graph features for a question and dispatch to one path
    or fuses both via the Synthesizer. Heuristic + deterministic (hermetic); an LLM router can
    replace route() behind the same signature later. Registers paths additively — it does not
    modify SqlPath or GraphRagPath."""

    name = "router"

    def __init__(
        self,
        sql_path: RetrievalPath,
        graph_path: RetrievalPath | None = None,
        synthesizer: Synthesizer | None = None,
        entity_vocab: object = None,
    ):
        self._sql = sql_path
        self._graph = graph_path
        self._synth = synthesizer or Synthesizer()
        self._entity_vocab = frozenset(e.lower() for e in (entity_vocab or ()))

    def _entity_named(self, words: set[str]) -> bool:
        for e in self._entity_vocab:
            etok = set(re.findall(r"[a-z0-9]+", e))
            if etok and etok <= words:
                return True
        return False

    def route(self, question: str) -> RoutingDecision:
        w = _words(question)
        sql_score = len(w & _SQL_HINTS)
        graph_score = len(w & _GRAPH_HINTS)
        if self._graph is None:
            return RoutingDecision(paths=("sql",), fuse=False)
        if sql_score > 0 and graph_score > 0:
            return RoutingDecision(paths=("sql", "graph"), fuse=True)
        if graph_score > sql_score:
            return RoutingDecision(paths=("graph",), fuse=False)
        if sql_score > graph_score:
            return RoutingDecision(paths=("sql",), fuse=False)
        # tie (typically both 0): a named entity tips toward the graph, else default to SQL
        if self._entity_named(w):
            return RoutingDecision(paths=("graph",), fuse=False)
        return RoutingDecision(paths=("sql",), fuse=False)

    def can_handle(self, question: str) -> bool:
        return True

    def run(self, question: str) -> RetrievalResult:
        decision = self.route(question)
        if decision.fuse:
            results = [self._sql.run(question), self._graph.run(question)]  # type: ignore[union-attr]
            return self._synth.fuse(question, results)
        if decision.paths == ("graph",) and self._graph is not None:
            return self._graph.run(question)
        return self._sql.run(question)
