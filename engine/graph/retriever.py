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

    def __init__(self, store: GraphStore, max_hops: int = 2):
        self._store = store
        self._max_hops = max_hops

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
            next_frontier: list[str] = []
            for ent in frontier:
                for t in self._store.neighbors(ent):
                    key = (t.subject, t.relation, t.obj, t.source)
                    if key not in seen_triples:
                        seen_triples.add(key)
                        collected.append(t)
                    for other in (t.subject, t.obj):
                        if other not in seen_entities:
                            seen_entities.add(other)
                            next_frontier.append(other)
            frontier = next_frontier
        return RetrievedSubgraph(seeds=tuple(seeds), triples=tuple(collected))
