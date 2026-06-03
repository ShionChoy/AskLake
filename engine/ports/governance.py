from __future__ import annotations

from typing import Protocol, runtime_checkable

from engine.ports.storage import QueryResult


@runtime_checkable
class GovernanceHook(Protocol):
    """Pre/post query governance. Passthrough (P0) -> RBAC/PII/cost guardrails (P3)."""

    def before_query(self, sql: str, role: str) -> str: ...

    def after_result(self, result: QueryResult, role: str) -> QueryResult: ...
