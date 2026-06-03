from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from engine.ports.storage import QueryResult


@dataclass
class RetrievalResult:
    path: str
    sql: str | None
    result: QueryResult | None
    narrative: str | None
    chart_spec: dict | None


@runtime_checkable
class RetrievalPath(Protocol):
    """A grounded retrieval path. SqlPath (P1), GraphRagPath (P4)."""

    name: str

    def can_handle(self, question: str) -> bool: ...

    def run(self, question: str) -> RetrievalResult: ...
