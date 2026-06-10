from __future__ import annotations

import re

from engine.graph.retriever import GraphRetriever, RetrievedSubgraph
from engine.ports.graph_store import GraphStore
from engine.ports.retrieval import RetrievalResult
from engine.ports.storage import QueryResult

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
    }
)


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _narrative(subgraph: RetrievedSubgraph) -> str:
    if not subgraph.triples:
        return "No matching facts found in the knowledge graph."
    seeds = ", ".join(subgraph.seeds) if subgraph.seeds else "the question"
    lines = [f"{t.subject} {t.relation} {t.obj} [{t.source}]" for t in subgraph.triples]
    return f"From the knowledge graph, starting at {seeds}:\n" + "\n".join(lines)


class GraphRagPath:
    """RetrievalPath (P4): answers from a knowledge graph via multi-hop retrieval with traceable
    citations. Additive sibling of SqlPath; selected by the Router. Reuses the generic
    RetrievalResult (facts ride in `result` as rows; citations in `narrative`)."""

    name = "graph"

    def __init__(
        self,
        store: GraphStore,
        max_hops: int = 2,
        *,
        attribute_relations: frozenset[str] = frozenset(),
        top_k_seeds: int = 10,
    ):
        self._store = store
        self._retriever = GraphRetriever(
            store,
            max_hops=max_hops,
            attribute_relations=attribute_relations,
            top_k_seeds=top_k_seeds,
        )

    def can_handle(self, question: str) -> bool:
        w = _words(question)
        if w & _GRAPH_HINTS:
            return True
        return bool(self._retriever.seeds(question))  # fast: inverted index, no full scan

    def run(self, question: str) -> RetrievalResult:
        sg = self._retriever.retrieve(question)
        rows = [(t.subject, t.relation, t.obj, t.source) for t in sg.triples]
        result = (
            QueryResult(columns=["subject", "relation", "object", "source"], rows=rows)
            if rows
            else None
        )
        return RetrievalResult(
            path=self.name,
            sql=None,
            result=result,
            narrative=_narrative(sg),
            chart_spec=None,
        )
