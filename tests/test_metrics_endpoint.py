from fastapi.testclient import TestClient

from api.main import create_app
from engine.lakehouse.duckdb_backend import DuckDBBackend
from engine.observability.prometheus import PrometheusObservability


def _seeded_backend():
    b = DuckDBBackend()
    b.setup("CREATE TABLE t AS SELECT 1 AS x;")
    return b


def test_metrics_present_with_prometheus_obs():
    obs = PrometheusObservability()
    client = TestClient(create_app(backend=_seeded_backend(), observability=obs))
    client.post("/query", json={"sql": "SELECT x FROM t LIMIT 1"})
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "asklake_spans_total" in r.text  # the /query span was recorded


def test_metrics_absent_with_default_noop():
    client = TestClient(create_app(backend=_seeded_backend()))
    assert client.get("/metrics").status_code == 404


def test_default_wiring_uses_prometheus_when_setting_enabled(monkeypatch):
    monkeypatch.setenv("ASKLAKE_OBSERVABILITY_BACKEND", "prometheus")
    client = TestClient(create_app(backend=_seeded_backend()))
    client.post("/query", json={"sql": "SELECT x FROM t LIMIT 1"})
    assert client.get("/metrics").status_code == 200


def test_metrics_absent_with_explicit_noop_arg():
    from engine.observability.noop import NoopObservability

    client = TestClient(create_app(backend=_seeded_backend(), observability=NoopObservability()))
    assert client.get("/metrics").status_code == 404
