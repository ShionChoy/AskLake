# engine/observability/noop.py
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager


class NoopObservability:
    """No-op metrics/trace sink. Replaced in P5."""

    @contextmanager
    def span(self, name: str, **attrs) -> Iterator[None]:
        yield

    def event(self, name: str, **fields) -> None:
        return None
