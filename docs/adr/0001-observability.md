# ADR 0001 — Observability via injected Prometheus registries and decorator adapters

## Status
Accepted (2026-06).

## Context
The `Observability` port (`span`/`event`) needs a production metrics adapter while keeping tests
isolated and allowing the application to run without a live metrics server.

## Decision
- Implement `PrometheusObservability` over `prometheus_client` with an **injected
  `CollectorRegistry`** — no global registry state, so unit tests are isolated and no server
  is required to produce metrics (`generate_latest(registry)` yields the exposition text).
- Instrument the rest of the system with **decorator adapters** (`ObservingLLMProvider`,
  `ObservingStorageBackend`) composed at assembly time rather than editing concrete adapters.
- Keep the default app no-op; make Prometheus opt-in via `ASKLAKE_OBSERVABILITY_BACKEND`,
  and expose `/metrics` only when the active observability has a `.registry`.
- Keep metrics infrastructure outside the repository. Operators can point their own Prometheus
  service at the application's `/metrics` endpoint.

## Consequences
- Real Prometheus metrics with zero changes to existing engine adapters; CI remains hermetic.
- Token-cost is NOT recorded yet: the `LLMProvider.complete` port returns only `str`, so no
  usage is available without threading it through. Documented as a carry-over (emit usage via
  `Observability.event` from inside the provider, without changing the port's return type).
