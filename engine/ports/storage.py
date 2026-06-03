from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class ColumnSchema:
    name: str
    type: str


@dataclass
class TableSchema:
    name: str
    columns: list[ColumnSchema]


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple[Any, ...]]


@runtime_checkable
class StorageBackend(Protocol):
    """Executes SQL and introspects schema. Implementations: DuckDB (P1), Iceberg (P1.5)."""

    def run_sql(self, sql: str) -> QueryResult: ...

    def list_tables(self) -> list[TableSchema]: ...
