from __future__ import annotations

from typing import Protocol, runtime_checkable

from engine.ports.storage import QueryResult


@runtime_checkable
class GovernanceHook(Protocol):
    """Pre/post-query governance interface for RBAC, PII, and cost guardrails."""

    def before_query(self, sql: str, role: str) -> str: ...

    def after_result(self, result: QueryResult, role: str) -> QueryResult: ...
