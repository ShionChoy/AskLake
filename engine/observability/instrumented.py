from __future__ import annotations

import contextlib

from engine.ports.llm import LLMProvider
from engine.ports.observability import Observability
from engine.ports.storage import QueryResult, StorageBackend, TableSchema


class ObservingLLMProvider:
    """Decorator-adapter: wraps any LLMProvider and emits spans/events to Observability.

    Additive — the wrapped provider and the LLMProvider port are untouched. Records call
    count + latency (the port returns only `str`, so no token usage is fabricated; real
    token accounting is a documented carry-over)."""

    def __init__(self, inner: LLMProvider, obs: Observability) -> None:
        self._inner = inner
        self._obs = obs

    def complete(self, prompt: str, system: str | None = None) -> str:
        with self._obs.span("llm.complete"):
            self._obs.event("llm_call")
            return self._inner.complete(prompt, system=system)


class ObservingStorageBackend:
    """Decorator-adapter: wraps any StorageBackend and emits spans/events.

    Records a `sql_error` event on failure, then re-raises (behavior preserved)."""

    def __init__(self, inner: StorageBackend, obs: Observability) -> None:
        self._inner = inner
        self._obs = obs

    def run_sql(self, sql: str) -> QueryResult:
        with self._obs.span("storage.run_sql"):
            try:
                return self._inner.run_sql(sql)
            except Exception:
                with contextlib.suppress(Exception):
                    self._obs.event("sql_error")
                raise

    def list_tables(self) -> list[TableSchema]:
        return self._inner.list_tables()
