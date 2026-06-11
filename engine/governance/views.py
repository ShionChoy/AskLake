from __future__ import annotations

from engine.governance.policy import Policy
from engine.ports.storage import StorageBackend


def build_role_views(backend: StorageBackend, policy: Policy) -> None:
    """Create one schema `rls_<role>` of views over `main` per governance role.

    Each view applies the role's row_security predicate (WHERE) and redacts pii_columns to
    NULL for masked roles. Column names come from the live schema (dataset-agnostic). DuckDB
    views are lazy, so this is cheap even over large parquet. Idempotent (CREATE OR REPLACE)."""
    tables = backend.list_tables()  # 'main' schema only
    for role in policy.roles:
        schema = f"rls_{role}"
        backend.run_sql(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
        masked = role in policy.mask_roles
        rs = policy.row_security.get(role, {})
        for t in tables:
            select_list = ", ".join(
                f'NULL AS "{c.name}"'
                if (masked and c.name in policy.pii_columns)
                else f'"{c.name}"'
                for c in t.columns
            )
            where = f" WHERE {rs[t.name]}" if t.name in rs else ""
            backend.run_sql(
                f'CREATE OR REPLACE VIEW "{schema}"."{t.name}" AS '
                f'SELECT {select_list} FROM main."{t.name}"{where}'
            )
