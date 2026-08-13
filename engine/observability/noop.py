# engine/observability/noop.py
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager


class NoopObservability:
    """No-op metrics and trace sink used when observability is disabled."""

    @contextmanager
    def span(self, name: str, **attrs) -> Iterator[None]:
        yield

    def event(self, name: str, **fields) -> None:
        return None
