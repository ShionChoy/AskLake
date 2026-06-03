from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Protocol, runtime_checkable


@runtime_checkable
class Observability(Protocol):
    """Metrics/trace seam. No-op (P0) -> Prometheus (P5)."""

    @contextmanager
    def span(self, name: str, **attrs) -> Iterator[None]: ...

    def event(self, name: str, **fields) -> None: ...
