from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Triple:
    """One knowledge-graph fact. `source` is the document id for citation/traceability."""

    subject: str
    relation: str
    obj: str
    source: str = ""


@runtime_checkable
class GraphStore(Protocol):
    """Knowledge-graph storage and traversal implemented in memory or with Neo4j."""

    def add(self, triple: Triple) -> None: ...

    def triples(self) -> tuple[Triple, ...]: ...

    def neighbors(self, entity: str) -> tuple[Triple, ...]: ...

    def entities(self) -> frozenset[str]: ...
