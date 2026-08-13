from __future__ import annotations

import re

from engine.governance.policy import GovernanceConfigurationError, Policy
from engine.ports.storage import StorageBackend

_SAFE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,62}$")


def _q(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _validate_column_rules(policy: Policy, role: str, tables: dict) -> None:
    role_tables = set(policy.role(role).tables)
    for reference in policy.role(role).columns:
        if "." not in reference:
            raise GovernanceConfigurationError(
                f"column rule {reference!r} must use table.column or *.column"
            )
        table_name, column_name = reference.split(".", 1)
        if table_name == "*":
            if tables and not any(
                column_name in {column.name for column in table.columns}
                for table in tables.values()
            ):
                raise GovernanceConfigurationError(
                    f"column rule {reference!r} does not match the live schema"
                )
            continue
        if "*" not in role_tables and table_name not in role_tables:
            raise GovernanceConfigurationError(
                f"column rule {reference!r} references a table not granted to role {role!r}"
            )
        table = tables.get(table_name)
        if table is not None and column_name not in {column.name for column in table.columns}:
            raise GovernanceConfigurationError(
                f"column rule {reference!r} does not match the live schema"
            )


def build_role_views(backend: StorageBackend, policy: Policy) -> None:
    """Materialize fail-closed row/column security views for every configured role.

    Generated schemas are rebuilt so removed permissions cannot survive as stale views. A bad
    predicate or column rule aborts startup; silently replacing it with a pass-through view would
    convert a configuration error into a data exposure.
    """

    live_tables = {table.name: table for table in backend.list_tables()}
    for name, table_policy in policy.table_policies.items():
        if table_policy.required and name not in live_tables:
            raise GovernanceConfigurationError(f"required governed table {name!r} is missing")

    for role in policy.roles:
        if not _SAFE_NAME.fullmatch(role):
            raise GovernanceConfigurationError(f"unsafe role name: {role!r}")
        _validate_column_rules(policy, role, live_tables)
        declared_tables = set(policy.role(role).tables)
        unknown_predicates = set(policy.role(role).row_security)
        if "*" not in declared_tables:
            unknown_predicates -= declared_tables
        else:
            unknown_predicates -= set(live_tables)
        if unknown_predicates:
            raise GovernanceConfigurationError(
                f"row-security rules for role {role!r} reference ungranted tables: "
                f"{sorted(unknown_predicates)}"
            )
        schema = f"rls_{role}"
        backend.run_sql(f"DROP SCHEMA IF EXISTS {_q(schema)} CASCADE")
        backend.run_sql(f"CREATE SCHEMA {_q(schema)}")
        allowed = policy.tables_for(role, set(live_tables))

        for table_name in sorted(allowed):
            table = live_tables[table_name]
            projections: list[str] = []
            for column in table.columns:
                handling = policy.column_handling(role, table_name, column.name)
                if handling == "deny":
                    continue
                if handling == "mask":
                    projections.append(f"CAST(NULL AS {column.type}) AS {_q(column.name)}")
                else:
                    projections.append(_q(column.name))
            if not projections:
                raise GovernanceConfigurationError(
                    f"role {role!r} has no visible columns in table {table_name!r}"
                )

            statement = (
                f"CREATE VIEW {_q(schema)}.{_q(table_name)} AS "
                f"SELECT {', '.join(projections)} FROM main.{_q(table_name)}"
            )
            predicate = policy.row_predicate(role, table_name)
            if predicate:
                statement += f" WHERE {predicate}"
            try:
                backend.run_sql(statement)
            except Exception as exc:  # noqa: BLE001 - converted to an explicit boot failure
                raise GovernanceConfigurationError(
                    f"could not build governed view for role={role!r}, table={table_name!r}"
                ) from exc
