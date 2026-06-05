from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest


class PrometheusObservability:
    """Observability adapter backed by prometheus_client (P5; fills the no-op P0 seam).

    Uses an *injected* CollectorRegistry so it is hermetic and unit-testable: no global
    registry state, no Prometheus server required. Scrape via `exposition()` (what the
    `/metrics` route serves) or point a Prometheus server at that route.
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        self._spans = Counter(
            "asklake_spans", "Spans entered, by name", ["name"], registry=self.registry
        )
        self._span_seconds = Histogram(
            "asklake_span_duration_seconds",
            "Span wall-clock duration in seconds, by name",
            ["name"],
            registry=self.registry,
        )
        self._events = Counter(
            "asklake_events", "Events emitted, by name", ["name"], registry=self.registry
        )

    @contextmanager
    def span(self, name: str, **attrs) -> Iterator[None]:
        # **attrs is intentionally NOT turned into Prometheus labels — arbitrary key/value
        # pairs would create unbounded label cardinality and blow up the metrics store.
        self._spans.labels(name=name).inc()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._span_seconds.labels(name=name).observe(time.perf_counter() - start)

    def event(self, name: str, **fields) -> None:
        # **fields is intentionally NOT turned into Prometheus labels — same cardinality reason.
        self._events.labels(name=name).inc()

    def exposition(self) -> bytes:
        """Prometheus text exposition format (what `/metrics` serves)."""
        return generate_latest(self.registry)
