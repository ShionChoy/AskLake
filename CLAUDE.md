# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What AskLake is

A governed, multi-agent natural-language analytics platform that answers questions over the user's own data through **two grounded retrieval paths** — `SqlPath` (Text-to-SQL over a lakehouse) and `GraphRagPath` (GraphRAG over a knowledge graph) — with a Router that picks one or fuses both. The LLM is a swappable component; the engineering value lives in everything around it (data platform, semantic layer, agent orchestration, governance, evaluation).

For the full architecture and roadmap, see `docs/AskLake-Plan.md`; for dataset specifics (IMDb + CMU, licenses, the IMDb↔CMU entity-alignment design), see `docs/dataset.md`.

## Cross-file architecture rules (these span many files — honor them)

- **Interface-first / additive / demo-never-regresses.** Seven ports anchor the design (`LLMProvider`, `StorageBackend`, `SchemaProvider`, `RetrievalPath`, `AgentGraph` nodes, `GovernanceHook`, `Observability`). Add a new capability by **adding an adapter or a graph node — never by rewriting** existing ones. Each capability ships a runnable `make demo-pX`, and CI smoke-tests **all prior demos** so nothing regresses.
- **Dataset-agnostic engine vs per-dataset config.** Code under `engine/` must **never hardcode IMDb column names**; it reads only from the connected schema + semantic layer. Everything dataset-specific (source connector, `semantic.yaml`, `governance.yaml`, graph ontology/prompts) lives under `datasets/<name>/`. This is also what makes the engine switchable / BYO.
- **Retrieval paths are pluggable behind `RetrievalPath`.** Adding `GraphRagPath` must not modify `SqlPath` or the Router core.

## Hard constraints

- **16 GB dev machine.** Use Docker Compose **profiles**; DuckDB (embedded) is the dev-time query engine. Spark / Trino / Neo4j are memory-heavy → start on-demand and shut down after.
- **LLM via cloud API by default, swappable** (`LLMProvider`; a local `OllamaProvider` is an optional add). Data never leaves the machine.
- **Never commit raw datasets** — use `scripts/download_data.sh`. IMDb is **non-commercial** (no redistribution); CMU is **CC BY-SA** (attribution + share-alike). Project license is Apache-2.0.

## Conventions & layout

- Python (uv or poetry) + ruff/black. `Makefile` targets: `make dev`, `make demo-pX` (and `make demo`), `make eval`, `make build-lake`.
- `docker compose --profile core up` for daily dev; add profiles `lake-build` (Spark/MinIO), `graph` (Neo4j), `observability`, `orchestration`, `query-dist` only when needed.
- `engine/ports/` (interfaces) + `engine/{llm,lakehouse,semantic,agents,retrieval,graph,governance,observability}/` (adapters) + `datasets/<name>/` (per-dataset config) + `api/ ui/ eval/ demos/ orchestration/ infra/ tests/smoke/ docs/`.
