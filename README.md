# AskLake

**Governed, multi-agent natural-language analytics over your own data.**

AskLake answers plain-English questions through two grounded retrieval paths — Text-to-SQL over a DuckDB/Parquet lakehouse and GraphRAG over a knowledge graph — with a Router that picks one or fuses both. The LLM is a swappable component (DeepSeek by default, Anthropic supported); the engineering value is everything around it: a port-and-adapter engine, a semantic layer, agentic self-correction, governance (RBAC/PII/cost guardrails), a quantified eval harness, and observability. Runs on a 16 GB laptop — DuckDB is embedded, no Docker required for local use.

---

## Features

- **Agentic Text-to-SQL** — a LangGraph pipeline (SchemaRetriever → SQLWriter → Validator → SelfCorrect ≤N) that executes the generated SQL and feeds errors back to the model for bounded self-correction.
- **Semantic layer** — curated table/column descriptions, metrics, synonyms (e.g. "score" → `averageRating`), and few-shot examples ground the LLM and eliminate whole categories of hallucinated column names.
- **GraphRAG second path** — multi-hop BFS over a knowledge graph with traceable citations; answers plot/theme questions that pure SQL cannot.
- **Heuristic Router + Synthesizer** — routes structured questions to SQL, relational/narrative questions to the graph, and fuses both for cross-cutting queries.
- **Governance hook** — RBAC, PII masking (`birthYear`/`deathYear` masked for `public` role), row-level filters (adult titles hidden), and query-cost guardrails (blocks unbounded scans and non-SELECT writes).
- **Observability** — `PrometheusObservability` records LLM-call latency, SQL errors, and storage spans; opt-in via `ASKLAKE_OBSERVABILITY_BACKEND=prometheus` which also exposes a `/metrics` endpoint.
- **Port-and-adapter engine** — 7 ports (`LLMProvider`, `StorageBackend`, `SchemaProvider`, `RetrievalPath`, `AgentGraph` nodes, `GovernanceHook`, `Observability`) keep every component swappable without touching existing adapters.
- **Dataset-agnostic** — the engine never hardcodes column names; everything dataset-specific (connector, `semantic.yaml`, `governance.yaml`, graph ontology) lives under `datasets/<name>/`.
- **Quantified eval harness** — execution-accuracy + valid-SQL-rate + self-correction count, with a 12-case hand-authored IMDb gold set and `make eval-real` for live LLM runs.

---

## Quickstart

### 1. Install

```bash
uv sync
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your key:
#   DEEPSEEK_API_KEY=sk-...
```

The API will boot and serve `/query` even without a key; only `/ask` (LLM-powered) requires one.

### 3. Download and build the data (one-time, ~1.8 GB)

```bash
bash scripts/download_data.sh   # downloads raw IMDb TSVs into ./data (gitignored)
make build-imdb                 # converts TSVs → Parquet; safe to re-run
```

### 4. Start the API (terminal 1)

```bash
make serve
# FastAPI on http://localhost:8000
# Endpoints: /ask  /query  /info  /ask_trace  /metrics
```

### 5. Start the UI (terminal 2)

```bash
make ui
# Streamlit on http://localhost:8501
```

Open **http://localhost:8501** in your browser.

### What you'll see

- A model caption (e.g. `model: deepseek-chat · semantic-grounded + self-correcting (agentic)`).
- An **Ask in natural language** box — type a question such as *"Highest-rated sci-fi films after 2010 (top 10)"*.
- The generated SQL, a **Backend processing steps** trace (schema retrieval → SQL generation → execution with ✅/❌ and timings; a red ❌ followed by a retry shows the self-correction loop in action), then a result table and bar chart.
- A **Raw SQL console** for direct DuckDB queries.

---

## How It Works

### Architecture

The engine under `engine/` is built around 7 ports with swappable adapters:

| Port | Default adapter |
|---|---|
| `LLMProvider` | `DeepSeekProvider` (OpenAI-compatible; `AnthropicProvider` is a drop-in) |
| `StorageBackend` | `DuckDBBackend` (embedded, 2 GB query memory cap) |
| `SchemaProvider` | `SemanticLayerProvider` (grounded; `RawSchemaProvider` kept as baseline) |
| `RetrievalPath` | `AgenticSqlPath` + `GraphRagPath`, dispatched by `Router` |
| `GovernanceHook` | `PolicyGovernance` (RBAC, PII masking, cost guardrails) |
| `Observability` | `PrometheusObservability` (opt-in; no-op by default) |

### Agentic Text-to-SQL

The LangGraph agent runs: **SchemaRetriever → SQLWriter → Validator → SelfCorrect (≤N) → END**. The Validator executes the SQL against DuckDB; on failure it injects the error message back into the model context. Example: the model first writes `SELECT title, rating FROM movies` (no such column); the validator's error (`Referenced column "rating" not found`) triggers a correction to `SELECT title, averageRating FROM movies`, which succeeds.

### GraphRAG

`GraphRagPath` does multi-hop BFS over a knowledge-graph triple store, tagging each hop with a source citation. The graph is built from an LLM entity/relation extraction pass constrained by a per-dataset ontology (`datasets/imdb_cmu/graph/ontology.yaml`). The default store is an in-process `InMemoryGraphStore`; a `Neo4jGraphStore` slots behind the same `GraphStore` port for larger graphs.

### Router

A heuristic Router scores SQL-vs-graph features and dispatches to `SqlPath`, `GraphRagPath`, or fuses both. A `Synthesizer` concatenates the SQL table with the graph narrative. Structured aggregation/filter questions route to SQL; plot/theme questions route to the graph; cross-cutting questions trigger fusion. All three components sit behind ports and can be replaced with LLM-backed implementations.

---

## Evaluation

Real run, DeepSeek `deepseek-chat` over a 12-case hand-authored IMDb gold set:

| system | n | valid-SQL | exec-accuracy | avg self-corrections |
|---|---|---|---|---|
| baseline (single-prompt)  | 12 | 92%  | 42% | 0.00 |
| agentic (self-correct)    | 12 | 100% | 50% | 0.08 |
| semantic layer (grounded) | 12 | 100% | 50% | 0.00 |

The self-correction loop lifts valid-SQL 92%→100% and execution accuracy 42%→50% over the naive baseline. The semantic layer reaches the same accuracy with zero self-corrections — grounding yields valid SQL on the first attempt. This is a 12-case real-data slice with strict multiset-exact scoring; the baseline→agentic delta is the signal.

Reproduce with `make eval-real` (requires a built parquet and an API key). A hermetic illustrative run (no key, no data) is available with `make eval`.

---

## Development

### Hermetic demos (no API key, no data download)

```bash
make demo        # runs demo-p0 through demo-p5 in sequence
make demo-p0     # DuckDB smoke test
make demo-p2     # agentic self-correction (rating → averageRating)
make demo-p3     # governance: analyst vs public role, cost guardrail
make demo-p4     # GraphRAG + Router fusion (Nolan films + plot themes)
make demo-p5     # observability: instrumented self-correction + /metrics excerpt
```

### Tests and lint

```bash
make test        # pytest unit + integration suite
make lint        # ruff check + ruff format --check
```

### CI

CI runs all demos (`p0`→`p5`), the full test suite, and a Docker image build on every push — no prior demo regresses.

### Clean up

```bash
make clean       # removes build artifacts and temporary files
```

---

## Configuration

| Variable | Purpose |
|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek auth (default provider) |
| `ANTHROPIC_API_KEY` + `ASKLAKE_LLM_PROVIDER=anthropic` | use Claude instead of DeepSeek |
| `ASKLAKE_PARQUET_DIR` | built parquet location (default `data/imdb/parquet`) |
| `ASKLAKE_OBSERVABILITY_BACKEND=prometheus` | enable `/metrics` Prometheus exposition |
| `ASKLAKE_API_PORT` | API port (default `8000`) |
| `ASKLAKE_API_URL` | URL the UI calls (default `http://localhost:8000`) |

### Switching LLM provider

DeepSeek is the default (fast, cheap, OpenAI-compatible). To use Claude:

```bash
# in .env
ANTHROPIC_API_KEY=sk-ant-...
ASKLAKE_LLM_PROVIDER=anthropic
```

Both providers implement the same `LLMProvider` port — no other config changes needed.

### Optional: Prometheus + Grafana

The observability stack is memory-heavy and off by default. Bring it up alongside the core stack when you want live dashboards:

```bash
docker compose --profile core --profile observability up
# Prometheus on :9090, Grafana on :3000 (anonymous admin)
# Dashboard: "AskLake Observability"
```

---

## Project Layout

```
engine/          # port interfaces + adapters (LLM, storage, semantic, agents, graph, governance, observability)
api/             # FastAPI app (/ask, /query, /info, /ask_trace, /metrics)
ui/              # Streamlit front-end
datasets/        # per-dataset config (semantic.yaml, governance.yaml, graph ontology, connector)
eval/            # eval harness + IMDb gold set (eval/imdb_gold.py)
demos/           # hermetic demo scripts (demo-p0 through demo-p5)
infra/           # Prometheus + Grafana provisioning (observability profile)
tests/           # unit + integration tests
```

---

## Data and License

**IMDb data** is non-commercial (Terms of Use prohibit redistribution). Raw TSV files are downloaded on demand via `bash scripts/download_data.sh` and written into `data/` (gitignored — never committed to this repo).

**CMU Movie Summary Corpus** is licensed CC BY-SA 4.0 (attribution + share-alike).

**Project code** is licensed **Apache-2.0**. See `LICENSE`.
