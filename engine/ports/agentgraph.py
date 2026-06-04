from __future__ import annotations

from typing import TypedDict

from engine.ports.storage import QueryResult


class GraphState(TypedDict, total=False):
    question: str
    schema_context: str
    sql: str
    result: QueryResult
    error: str
    narrative: str
    chart_spec: dict
    attempts: int
