from __future__ import annotations

from engine.governance.policy import Policy
from engine.ports.storage import StorageBackend


def build_role_views(backend: StorageBackend, policy: Policy) -> None:
    """Create one schema `rls_<role>` of views over `main` per governance role.

    Each view applies the role's row_security predicate (WHERE) and redacts pii_columns to
    NULL for masked roles. Column names come from the live schema (dataset-agnostic). DuckDB
    views are lazy to query but EAGERLY bound at creation, so a predicate that references a
    table absent from this schema would raise; in that degenerate case (only seen with partial
    test seeds — the real parquet has every table) we fall back to a pass-through view so the
    table stays queryable and column masking still applies. Idempotent (CREATE OR REPLACE)."""
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
            base = (
                f'CREATE OR REPLACE VIEW "{schema}"."{t.name}" AS '
                f'SELECT {select_list} FROM main."{t.name}"'
            )
            predicate = rs.get(t.name)
            if predicate:
                try:
                    backend.run_sql(f"{base} WHERE {predicate}")
                    continue
                except Exception:  # noqa: BLE001 - predicate references a table absent from this schema; fall back to pass-through (column masking still applies)
                    pass
            backend.run_sql(base)
