from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError
from sqlglot.optimizer.scope import Scope, traverse_scope


class SqlPolicyError(ValueError):
    """A SQL statement cannot be safely executed under the query policy."""

    def __init__(self, message: str, *, code: str = "invalid_sql") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class SqlGuardrails:
    max_length: int = 20_000
    max_tables: int = 8
    max_joins: int = 7
    max_rows: int = 500
    require_limit: bool = False
    forbid_cross_join: bool = True
    forbidden_functions: frozenset[str] = frozenset(
        {
            "CSV_SCAN",
            "CURRENT_SETTING",
            "FILE_MODIFIED_TIME",
            "FILE_SIZE",
            "GETENV",
            "GLOB",
            "HTTP_GET",
            "HTTP_POST",
            "PARQUET_SCAN",
            "QUERY",
            "QUERY_TABLE",
            "READ_BLOB",
            "READ_CSV",
            "READ_CSV_AUTO",
            "READ_JSON",
            "READ_JSON_AUTO",
            "READ_NDJSON",
            "READ_PARQUET",
            "READ_TEXT",
            "SQLITE_SCAN",
            "POSTGRES_SCAN",
            "WHICH_SECRET",
        }
    )


@dataclass(frozen=True)
class QueryAnalysis:
    expression: exp.Expression
    tables: tuple[str, ...]
    table_nodes: tuple[exp.Table, ...]


def _physical_tables(tree: exp.Expression) -> tuple[exp.Table, ...]:
    """Resolve tables by lexical scope so a nested CTE cannot shadow an outer physical table."""

    found: dict[int, exp.Table] = {}
    for scope in traverse_scope(tree):
        for source in scope.sources.values():
            if isinstance(source, exp.Table):
                found[id(source)] = source
            elif not isinstance(source, Scope):
                # Unknown source kinds are denied rather than assumed to be harmless aliases.
                raise SqlPolicyError("query source cannot be safely resolved")
    return tuple(found.values())


def _function_name(function: exp.Func) -> str:
    # DuckDB extension/generic functions parse as Anonymous; sql_name() then returns ANONYMOUS,
    # so the identifier itself is the security-relevant name.
    if isinstance(function, exp.Anonymous):
        return function.name.upper()
    return function.sql_name().upper()


def analyze_read_query(sql: str, guardrails: SqlGuardrails) -> QueryAnalysis:
    """Parse and validate one bounded, read-only DuckDB query.

    This is intentionally an allow-list: only SELECT/set-operation roots are accepted. SQLGlot
    may represent unsupported syntax as ``Command`` instead of raising, so checking the root type
    is a required security boundary rather than a convenience validation.
    """

    if not isinstance(sql, str) or not sql.strip():
        raise SqlPolicyError("query is empty")
    if len(sql) > guardrails.max_length:
        raise SqlPolicyError(
            f"query exceeds the {guardrails.max_length}-character limit",
            code="query_too_large",
        )
    try:
        statements = [s for s in sqlglot.parse(sql, read="duckdb") if s is not None]
    except ParseError as exc:
        raise SqlPolicyError("query could not be parsed safely") from exc
    if len(statements) != 1:
        raise SqlPolicyError("only one SQL statement is permitted", code="multiple_statements")

    tree = statements[0]
    if not isinstance(tree, (exp.Select, exp.SetOperation)):
        raise SqlPolicyError(
            "only read-only SELECT queries are permitted", code="statement_not_read_only"
        )
    if tree.find(exp.Into) is not None:
        raise SqlPolicyError("SELECT INTO is not permitted", code="statement_not_read_only")
    with_clause = tree.args.get("with_")
    if with_clause is not None and with_clause.args.get("recursive"):
        raise SqlPolicyError(
            "recursive common-table expressions are not permitted",
            code="query_too_complex",
        )

    physical_tables = _physical_tables(tree)
    for table in physical_tables:
        # Table-valued functions (read_parquet, duckdb_secrets, range, …) can cross the governed
        # catalog boundary. They are denied even when the function is not on the explicit list.
        if not isinstance(table.this, exp.Identifier):
            raise SqlPolicyError(
                "table-valued functions are not permitted", code="external_access_denied"
            )
        if table.db or table.catalog:
            raise SqlPolicyError(
                "catalog- or schema-qualified tables are not permitted",
                code="schema_access_denied",
            )
    table_names = tuple(dict.fromkeys(table.name for table in physical_tables))
    if len(physical_tables) > guardrails.max_tables:
        raise SqlPolicyError(
            f"query references more than {guardrails.max_tables} tables",
            code="query_too_complex",
        )

    joins = list(tree.find_all(exp.Join))
    if len(joins) > guardrails.max_joins:
        raise SqlPolicyError(
            f"query contains more than {guardrails.max_joins} joins",
            code="query_too_complex",
        )
    if guardrails.forbid_cross_join:
        for join in joins:
            is_cross = (join.kind or "").upper() == "CROSS"
            is_comma_or_unqualified = not any(
                join.args.get(key) is not None for key in ("on", "using", "method")
            )
            if is_cross or is_comma_or_unqualified:
                raise SqlPolicyError(
                    "cross joins and joins without a condition are not permitted",
                    code="cross_join_denied",
                )

    forbidden = {name.upper() for name in guardrails.forbidden_functions}
    for function in tree.find_all(exp.Func):
        function_name = _function_name(function)
        unsafe_prefix = function_name.startswith(
            ("HTTP_", "PARQUET_", "POSTGRES_", "READ_", "S3_", "SQLITE_")
        )
        if function_name in forbidden or unsafe_prefix:
            raise SqlPolicyError(
                f"function {function_name} is not permitted",
                code="external_access_denied",
            )

    if guardrails.require_limit and tree.args.get("limit") is None:
        raise SqlPolicyError("an outer LIMIT clause is required", code="result_limit_required")
    return QueryAnalysis(expression=tree, tables=table_names, table_nodes=physical_tables)


def scope_read_query(
    sql: str,
    *,
    schema: str,
    allowed_tables: frozenset[str],
    guardrails: SqlGuardrails,
    max_rows: int | None = None,
) -> tuple[str, tuple[str, ...]]:
    """Validate a query, authorize its tables, and bind every table to one role schema."""

    analysis = analyze_read_query(sql, guardrails)
    allowed = {name.casefold() for name in allowed_tables}
    for table in analysis.table_nodes:
        if not isinstance(table.this, exp.Identifier):
            # analyze_read_query already rejects this; retain a fail-closed invariant here.
            raise SqlPolicyError("table-valued functions are not permitted")
        if table.name.casefold() not in allowed:
            raise SqlPolicyError(
                f"access to table {table.name!r} is not permitted", code="table_access_denied"
            )
        table.set("catalog", None)
        table.set("db", exp.to_identifier(schema, quoted=True))

    scoped = analysis.expression.sql(dialect="duckdb")
    cap = max_rows if max_rows is not None else guardrails.max_rows
    if cap <= 0:
        raise SqlPolicyError("configured result limit must be positive", code="policy_invalid")
    # An outer cap is a non-bypassable response-size guard even when a caller supplies a larger
    # inner LIMIT. It deliberately preserves the semantics of ORDER BY, aggregates, and set ops.
    scoped = f'SELECT * FROM ({scoped}) AS "_asklake_governed" LIMIT {int(cap)}'
    return scoped, analysis.tables
