from __future__ import annotations

import re
from dataclasses import dataclass

from engine.ports.graph_store import GraphStore, Triple

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


@dataclass(frozen=True)
class RetrievedSubgraph:
    seeds: tuple[str, ...]
    triples: tuple[Triple, ...]


class GraphRetriever:
    """Multi-hop retrieval over a GraphStore: find seed entities named in the question, BFS-expand
    up to `max_hops`, and return the connected subgraph (triples carry source citations). Lexical
    seed-matching is the hermetic stand-in for an embedding-based entity linker."""

    def __init__(
        self,
        store: GraphStore,
        max_hops: int = 2,
        *,
        max_triples: int = 300,
        max_neighbors_per_node: int = 50,
        max_degree: int = 100,
    ):
        self._store = store
        self._max_hops = max_hops
        self._max_triples = max_triples
        self._max_neighbors_per_node = max_neighbors_per_node
        self._max_degree = max_degree

    def _seeds(self, question: str) -> list[str]:
        qtok = _tokens(question)
        seeds = []
        for e in self._store.entities():
            etok = _tokens(e)
            if etok and etok <= qtok:  # every token of the entity name appears in the question
                seeds.append(e)
        return seeds

    def retrieve(self, question: str) -> RetrievedSubgraph:
        seeds = self._seeds(question)
        seen_entities: set[str] = set(seeds)
        seen_triples: set[tuple[str, str, str, str]] = set()
        collected: list[Triple] = []
        frontier = list(seeds)
        for _ in range(self._max_hops):
            if len(collected) >= self._max_triples:
                break
            next_frontier: list[str] = []
            for ent in frontier:
                if len(collected) >= self._max_triples:
                    break
                nbrs = self._store.neighbors(ent)
                expandable = len(nbrs) <= self._max_degree  # don't traverse through hubs
                for t in nbrs[: self._max_neighbors_per_node]:  # cap per-node fan-out
                    key = (t.subject, t.relation, t.obj, t.source)
                    if key not in seen_triples:
                        seen_triples.add(key)
                        collected.append(t)
                        if len(collected) >= self._max_triples:
                            break
                    if expandable:
                        for other in (t.subject, t.obj):
                            if other not in seen_entities:
                                seen_entities.add(other)
                                next_frontier.append(other)
            frontier = next_frontier
        return RetrievedSubgraph(seeds=tuple(seeds), triples=tuple(collected))
