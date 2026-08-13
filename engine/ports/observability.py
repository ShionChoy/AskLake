from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol, runtime_checkable


@runtime_checkable
class Observability(Protocol):
    """Metrics and tracing interface implemented by no-op and Prometheus adapters."""

    @contextmanager
    def span(self, name: str, **attrs) -> Iterator[None]: ...

    def event(self, name: str, **fields) -> None: ...
