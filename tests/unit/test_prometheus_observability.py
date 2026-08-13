"""Unit tests for the Prometheus observability adapter."""

from engine.observability.prometheus import PrometheusObservability
from engine.ports.observability import Observability


def test_satisfies_observability_port():
    assert isinstance(PrometheusObservability(), Observability)


def test_span_counts_and_times():
    obs = PrometheusObservability()
    with obs.span("query", role="analyst"):
        pass
    assert obs.registry.get_sample_value("asklake_spans_total", {"name": "query"}) == 1.0
    assert (
        obs.registry.get_sample_value("asklake_span_duration_seconds_count", {"name": "query"})
        == 1.0
    )


def test_event_counts_and_exposition():
    obs = PrometheusObservability()
    obs.event("query_error", error="boom")
    obs.event("query_error", error="boom2")
    assert obs.registry.get_sample_value("asklake_events_total", {"name": "query_error"}) == 2.0
    body = obs.exposition()
    assert b"asklake_events_total" in body


def test_injected_registry_isolates_state():
    a, b = PrometheusObservability(), PrometheusObservability()
    a.event("x")
    assert a.registry.get_sample_value("asklake_events_total", {"name": "x"}) == 1.0
    assert b.registry.get_sample_value("asklake_events_total", {"name": "x"}) is None


def test_span_records_duration_even_when_body_raises():
    import pytest

    obs = PrometheusObservability()
    with pytest.raises(ValueError):
        with obs.span("risky"):
            raise ValueError("boom")
    assert (
        obs.registry.get_sample_value("asklake_span_duration_seconds_count", {"name": "risky"})
        == 1.0
    )
    assert obs.registry.get_sample_value("asklake_spans_total", {"name": "risky"}) == 1.0
