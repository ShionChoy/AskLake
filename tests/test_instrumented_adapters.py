import pytest

from engine.lakehouse.duckdb_backend import DuckDBBackend
from engine.llm.fake import FakeLLMProvider
from engine.observability.instrumented import (
    ObservingLLMProvider,
    ObservingStorageBackend,
)
from engine.observability.prometheus import PrometheusObservability
from engine.ports.llm import LLMProvider
from engine.ports.storage import StorageBackend


def test_observing_llm_satisfies_port_and_counts_calls():
    obs = PrometheusObservability()
    llm = ObservingLLMProvider(FakeLLMProvider(responses=["a", "b"]), obs)
    assert isinstance(llm, LLMProvider)
    assert llm.complete("q1") == "a"
    assert llm.complete("q2", system="s") == "b"
    assert obs.registry.get_sample_value("asklake_events_total", {"name": "llm_call"}) == 2.0
    assert (
        obs.registry.get_sample_value(
            "asklake_span_duration_seconds_count", {"name": "llm.complete"}
        )
        == 2.0
    )


def test_observing_storage_satisfies_port_and_passes_results_through():
    obs = PrometheusObservability()
    inner = DuckDBBackend()
    inner.setup("CREATE TABLE t AS SELECT 1 AS x;")
    backend = ObservingStorageBackend(inner, obs)
    assert isinstance(backend, StorageBackend)
    res = backend.run_sql("SELECT x FROM t")
    assert res.rows == [(1,)]
    assert backend.list_tables()  # delegated, non-empty
    assert (
        obs.registry.get_sample_value(
            "asklake_span_duration_seconds_count", {"name": "storage.run_sql"}
        )
        == 1.0
    )


def test_observing_storage_records_sql_error_and_reraises():
    obs = PrometheusObservability()
    inner = DuckDBBackend()
    inner.setup("CREATE TABLE t AS SELECT 1 AS x;")
    backend = ObservingStorageBackend(inner, obs)
    with pytest.raises(Exception):  # noqa: B017 - any backend error must propagate
        backend.run_sql("SELECT nope FROM t")
    assert obs.registry.get_sample_value("asklake_events_total", {"name": "sql_error"}) == 1.0
