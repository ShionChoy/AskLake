# engine/governance/passthrough.py
from __future__ import annotations

from engine.ports.storage import QueryResult


class PassthroughGovernance:
    """No-op governance adapter that returns inputs unchanged."""

    def before_query(self, sql: str, role: str) -> str:
        return sql

    def after_result(self, result: QueryResult, role: str) -> QueryResult:
        return result
