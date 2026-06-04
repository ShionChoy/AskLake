from __future__ import annotations

from collections import defaultdict

from engine.ports.graph_store import Triple


class InMemoryGraphStore:
    """Dependency-free in-process knowledge graph (hermetic stand-in for Neo4j). Stores triples
    plus an undirected adjacency index (entity -> touching triples) for multi-hop traversal.
    A Neo4jGraphStore can implement the same GraphStore port later without touching callers."""

    def __init__(self) -> None:
        self._triples: list[Triple] = []
        self._adj: dict[str, list[Triple]] = defaultdict(list)

    def add(self, triple: Triple) -> None:
        self._triples.append(triple)
        self._adj[triple.subject].append(triple)
        self._adj[triple.obj].append(triple)

    def triples(self) -> tuple[Triple, ...]:
        return tuple(self._triples)

    def neighbors(self, entity: str) -> tuple[Triple, ...]:
        return tuple(self._adj.get(entity, ()))

    def entities(self) -> frozenset[str]:
        out: set[str] = set()
        for t in self._triples:
            out.add(t.subject)
            out.add(t.obj)
        return frozenset(out)
