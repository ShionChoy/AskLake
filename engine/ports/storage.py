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
    """Execute SQL and introspect schema. DuckDB is the default implementation."""

    def run_sql(self, sql: str) -> QueryResult: ...

    def list_tables(self) -> list[TableSchema]: ...
