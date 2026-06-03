# engine/lakehouse/duckdb_backend.py
from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb

from engine.ports.storage import ColumnSchema, QueryResult, TableSchema


def _to_python(value: Any) -> Any:
    """Coerce DuckDB-specific types to standard Python primitives.

    DuckDB 1.5.x infers DECIMAL for float-looking VALUES literals and returns
    ``decimal.Decimal`` objects.  Callers and tests expect plain ``float``.
    """
    if isinstance(value, Decimal):
        return float(value)
    return value


class DuckDBBackend:
    """StorageBackend over DuckDB. In-memory by default; can register a directory of
    parquet files as views (one view per file, named by file stem)."""

    def __init__(self, database: str = ":memory:", parquet_dir: str | None = None):
        self._con = duckdb.connect(database)
        if parquet_dir:
            for pq in sorted(Path(parquet_dir).glob("*.parquet")):
                view = pq.stem
                # Parameter binding is not supported in CREATE VIEW DDL in DuckDB 1.5.x;
                # the path is a local filesystem path under our control (no injection risk).
                self._con.execute(
                    f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM read_parquet('{pq}')"
                )

    def setup(self, sql: str) -> None:
        """Run DDL/setup statements (e.g., seed tables). Not part of the port."""
        self._con.execute(sql)

    def run_sql(self, sql: str) -> QueryResult:
        cur = self._con.execute(sql)
        columns = [d[0] for d in cur.description] if cur.description else []
        rows = [tuple(_to_python(v) for v in r) for r in cur.fetchall()]
        return QueryResult(columns=columns, rows=rows)

    def list_tables(self) -> list[TableSchema]:
        rows = self._con.execute(
            """
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'main'
            ORDER BY table_name, ordinal_position
            """
        ).fetchall()
        by_table: dict[str, list[ColumnSchema]] = {}
        for table_name, column_name, data_type in rows:
            by_table.setdefault(table_name, []).append(
                ColumnSchema(name=column_name, type=data_type)
            )
        return [TableSchema(name=name, columns=cols) for name, cols in by_table.items()]
