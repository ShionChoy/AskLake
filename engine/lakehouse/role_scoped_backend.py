from __future__ import annotations

from engine.governance.policy import GovernanceError, Policy
from engine.governance.sql import SqlPolicyError, scope_read_query
from engine.ports.storage import QueryResult, StorageBackend, TableSchema


class RoleScopedBackend:
    """Storage facade that binds every logical table to immutable per-role views.

    Unlike a connection-level ``search_path``, AST rewriting is request-local: concurrent callers
    cannot change each other's role, and explicit references such as ``main.title_basics`` or
    ``rls_analyst.title_basics`` are rejected before DuckDB sees the query.
    """

    def __init__(
        self,
        base: StorageBackend,
        role: str,
        *,
        policy: Policy | None = None,
        action: str = "ask",
    ) -> None:
        self._base = base
        self._role = role
        self._policy = policy or Policy(roles=(role,))
        self._action = action
        self._schema = f"rls_{role}"

        available = {table.name for table in base.list_tables()}
        self._allowed = self._policy.tables_for(role, available)

    def run_sql(self, sql: str) -> QueryResult:
        if not self._policy.allows_action(self._role, self._action):
            raise GovernanceError(
                f"role {self._role!r} cannot perform {self._action!r}", code="action_denied"
            )
        try:
            scoped, _ = scope_read_query(
                sql,
                schema=self._schema,
                allowed_tables=self._allowed,
                guardrails=self._policy.guardrails,
                max_rows=self._policy.max_rows_for(self._role),
            )
        except SqlPolicyError as exc:
            raise GovernanceError(str(exc), code=exc.code) from exc
        return self._base.run_sql(scoped)

    def list_tables(self) -> list[TableSchema]:
        return [table for table in self._base.list_tables() if table.name in self._allowed]
