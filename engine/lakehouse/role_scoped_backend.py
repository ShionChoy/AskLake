from __future__ import annotations

from engine.ports.storage import QueryResult, StorageBackend, TableSchema


class RoleScopedBackend:
    """StorageBackend facade that runs each query under a role's `rls_<role>` schema.

    Sets `search_path` immediately before delegating, so unqualified table names resolve to
    the role's filtered/redacted views. The SET is issued by this facade (not user SQL), so it
    bypasses governance's read-only guard. Connection-scoped search_path is safe under the
    single-user, local-first assumption (no concurrent cross-role requests)."""

    def __init__(self, base: StorageBackend, role: str):
        self._base = base
        self._role = role

    def run_sql(self, sql: str) -> QueryResult:
        self._base.run_sql(f"SET search_path='rls_{self._role}'")
        return self._base.run_sql(sql)

    def list_tables(self) -> list[TableSchema]:
        return self._base.list_tables()
