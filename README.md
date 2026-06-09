# AskLake

**Governed, multi-agent natural-language analytics over your own data.**

AskLake answers plain-English questions through two grounded retrieval paths — Text-to-SQL over a DuckDB/Parquet lakehouse and GraphRAG over a knowledge graph — with a Router that picks one or fuses both. The LLM is a swappable component (DeepSeek by default, Anthropic supported); the engineering value is everything around it: a port-and-adapter engine, a semantic layer, agentic self-correction, governance (RBAC/PII/cost guardrails), a quantified eval harness, and observability. Runs on a 16 GB laptop — DuckDB is embedded, no Docker required for local use.

---

## Features

- **Grounded Text-to-SQL** — the default LangGraph pipeline (schema-link + value-link → classify → plan → write K candidates → validate + self-consistency → critic → self-correct ≤N) grounds the model in real schema **and stored values**, samples multiple candidates and takes the majority result, and verifies the answer for *correctness*, not just executability. (The simpler self-correct-only `AgenticSqlPath` is kept as an additive sibling / eval baseline; `ASKLAKE_AGENT=agentic` selects it.)
- **Semantic layer** — curated table/column descriptions, metrics, synonyms (e.g. "score" → `averageRating`), and few-shot examples ground the LLM and eliminate whole categories of hallucinated column names.
- **GraphRAG second path** — multi-hop BFS over a knowledge graph with traceable citations; answers plot/theme questions that pure SQL cannot.
- **Interactive network view** — graph and fusion answers also render their triples as a draggable, zoomable pyvis network (relation-labeled edges, hubs sized by degree, source citations on hover) in a 🕸️ Network view expander beneath the table. Adjustable in-app: node size, layout spacing/density, and a freeze-layout toggle.
- **Heuristic Router + Synthesizer** — routes structured questions to SQL, relational/narrative questions to the graph, and fuses both for cross-cutting queries.
- **Governance hook** — RBAC, PII masking (`birthYear`/`deathYear` masked for `public` role), row-level filters (adult titles hidden), and query-cost guardrails (blocks unbounded scans and non-SELECT writes).
- **Observability** — `PrometheusObservability` records LLM-call latency, SQL errors, and storage spans; opt-in via `ASKLAKE_OBSERVABILITY_BACKEND=prometheus` which also exposes a `/metrics` endpoint.
- **Bring-your-own key in the browser** — paste an API key and pick the provider/model right in the UI sidebar, then optionally save it to a local `0600` file for next time (or delete it). Credentials are sent per request and **never persisted server-side**; the server boots fine with no key at all.
- **Port-and-adapter engine** — 7 ports (`LLMProvider`, `StorageBackend`, `SchemaProvider`, `RetrievalPath`, `AgentGraph` nodes, `GovernanceHook`, `Observability`) keep every component swappable without touching existing adapters.
- **Dataset-agnostic** — the engine never hardcodes column names; everything dataset-specific (connector, `semantic.yaml`, `governance.yaml`, graph ontology) lives under `datasets/<name>/`.
- **Quantified eval harness** — a five-rung ablation (baseline → +semantic → +value-link → +plan/self-consistency → grounded) reporting execution-accuracy, valid-SQL-rate, and cost columns per difficulty tier, run on a 102-case tie-safe IMDb gold set **and** a synthetic CRM second dataset (a generalization proof: same engine, no hand-authored few-shots) via `make eval-real` / `make eval-real-crm`.

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

Providing the key here is **optional** — you can instead paste it in the browser sidebar at runtime (and save it locally for next time). The API boots and serves `/query` even with no key configured; only LLM-powered questions need one, supplied either way.

### 3. Download and build the data (one-time, ~1.8 GB)

```bash
bash scripts/download_data.sh   # downloads raw IMDb TSVs into ./data (gitignored)
make build-imdb                 # converts TSVs → Parquet; safe to re-run
```

### 3b. (Optional) Build the knowledge graph for GraphRAG

```bash
GRAPH_FILMS=200 make build-graph    # one-time LLM extraction over CMU plots; needs your key
# writes data/imdb/graph/triples.jsonl; the API loads it automatically at boot
```

Without this, the browser still answers SQL questions; graph/fusion routing activates once the graph is built.

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

- A **⚙️ Model & API key** sidebar — choose the provider and model, paste your API key (masked), and **Save locally** / **Delete saved key**. Ask before entering a key and you'll get a friendly "enter your API key in the sidebar" prompt rather than an error.
- A **Retrieval path** selector (Auto / SQL / Graph / Fusion). Auto routes structured questions to SQL and plot/theme questions to the graph (fusing both when relevant); Graph answers need no API key. Graph/fusion answers show a cited narrative (each fact tagged with its source plot).
- A model caption (e.g. `model: deepseek-v4-flash · provider: deepseek`) reflecting your selection.
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
| `RetrievalPath` | `GroundedSqlPath` + `GraphRagPath`, dispatched by `Router` (`AgenticSqlPath` kept as baseline; `ASKLAKE_AGENT=agentic` selects it) |
| `GovernanceHook` | `PolicyGovernance` (RBAC, PII masking, cost guardrails) |
| `Observability` | `PrometheusObservability` (opt-in; no-op by default) |

### Grounded Text-to-SQL

The default agent (`GroundedSqlPath`) runs: **link (schema + value-linking) → classify (difficulty) → plan (hard questions) → write K candidates → validate + self-consistency → critic → self-correct (≤N) → END**. *link* grounds the model in the semantic layer **and real stored values** (the question token "sci fi" is pinned to `genres LIKE '%Sci-Fi%'`; a name is resolved against the column to its canonical spelling); *write* samples K candidates for hard (top-N / multi-hop) questions, and *validate + self-consistency* executes them all and takes the **majority result set**, discarding queries that run but return different (wrong) rows; the *critic* checks the chosen result for correctness (0 rows, missing `ORDER BY`/`LIMIT`, …) and only then triggers a bounded self-correction.

The simpler self-correct-only path (`AgenticSqlPath`: **SQLWriter → Validator → SelfCorrect ≤N**, retrying only on execution *errors*) is kept as an additive sibling and the eval baseline. Example: the model first writes `SELECT title, rating FROM movies` (no such column); the validator's error (`Referenced column "rating" not found`) triggers a correction to `SELECT title, averageRating FROM movies`. As the eval shows, this loop fixes *executability* but not *correctness* — grounding does that.

### GraphRAG

`GraphRagPath` does multi-hop BFS over a knowledge-graph triple store, tagging each hop with a source citation. The graph is built from an LLM entity/relation extraction pass constrained by a per-dataset ontology (`datasets/imdb_cmu/graph/ontology.yaml`). The default store is an in-process `InMemoryGraphStore`; a `Neo4jGraphStore` slots behind the same `GraphStore` port for larger graphs.

### Router

A heuristic Router scores SQL-vs-graph features and dispatches to `SqlPath`, `GraphRagPath`, or fuses both. A `Synthesizer` concatenates the SQL table with the graph narrative. Structured aggregation/filter questions route to SQL; plot/theme questions route to the graph; cross-cutting questions trigger fusion. All three components sit behind ports and can be replaced with LLM-backed implementations.

---

## Evaluation

A five-rung **ablation** — each rung adds one capability over the previous — run live with DeepSeek `deepseek-v4-flash`, strict multiset-exact execution accuracy with tie-safe gold, reported per difficulty tier with cost columns (`llm/q` = mean LLM calls per question, `ms/q` = mean wall-clock). Run on **two datasets with the same engine and zero engine changes** — only `datasets/<name>/` config differs.

**IMDb** — 102-case stratified gold set on a ~243K-movie working set (`make build-imdb MIN_VOTES=25`):

| system | valid-SQL | exec-acc | llm/q | ms/q | aggregation | top-N | multi-hop |
|---|---|---|---|---|---|---|---|
| baseline (raw, single-prompt) | 98%  | 38% | 1.0 | 5200 | 66% | 24% | 19% |
| +semantic (grounded + self-correct) | 100% | **73%** | 1.0 | 5600 | 66% | 61% | 94% |
| +value-link | 100% | 72% | 1.0 | 5600 | 66% | 64% | 87% |
| +plan/self-consistency | 100% | 74% | 4.0 | 16900 | 66% | 67% | 90% |
| grounded (+ critic) | 100% | 75% | 4.2 | 16100 | 66% | 67% | 94% |

**CRM** — a synthetic second dataset (`datasets/crm_demo/`, 11-case gold) the engine was **never tuned for, with no hand-authored few-shots**:

| system | valid-SQL | exec-acc | llm/q | ms/q | aggregation | top-N | multi-hop |
|---|---|---|---|---|---|---|---|
| baseline (raw, single-prompt) | 100% | 55% | 1.0 | 3800 | 75% | 50% | 40% |
| +semantic | 100% | 91% | 1.0 | 2300 | 100% | 50% | 100% |
| +value-link | 100% | 82% | 1.0 | 2300 | 100% | 50% | 80% |
| +plan/self-consistency | 100% | 82% | 1.8 | 5100 | 100% | 50% | 80% |
| grounded (+ critic) | 100% | **100%** | 1.8 | 4300 | 100% | 100% | 100% |

**Grounding is the decisive lever on both datasets.** A semantic layer (curated descriptions, metrics, synonyms) lifts overall execution accuracy **38% → 73%** on IMDb and **55% → 91%** on CRM, concentrated on the hard tiers (IMDb multi-hop 19% → 94%, top-N 24% → 61%) and flat on aggregation (66% throughout — it needs no domain mapping). Self-correction in that step buys **executability** (valid-SQL 98% → 100%), not accuracy: the loop only retries on execution *errors*, never on a query that runs yet returns wrong rows.

**The heavier machinery earns its keep where there are no few-shots.** On hand-tuned IMDb, value-linking, planner decomposition, and self-consistency are roughly accuracy-neutral over the semantic layer (73% → 75%) while costing **4× the LLM calls and ~3× the latency** — the curated few-shots already carry it. But on **CRM, which has no few-shots**, the full grounded path is what reaches **100%** (vs 91% for the semantic layer alone): the reflexion critic closes the top-N tier 50% → 100%. That is the generalization payoff — the mechanisms that look redundant on a tuned dataset are exactly what get the last mile on a brand-new one, with the cost made explicit so the trade-off is visible.

Caveats (honest): CRM n=11 is small, so the intermediate rungs are noisy (the robust signal is baseline ≪ grounded); both are single live runs; K candidates are generated sequentially in this version (the `ms/q` columns reflect that — parallel fan-out is a latency follow-up that does not change accuracy).

Reproduce: `make build-imdb MIN_VOTES=25 && make eval-real` (IMDb) and `make build-crm && make eval-real-crm` (CRM) — both need an API key; IMDb also needs the raw TSVs. A hermetic illustrative run (no key, no data) is available with `make eval`.

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

> For local use you don't have to set any key/model variable: the browser sidebar (**⚙️ Model & API key**) lets you paste a key and pick a provider/model, and optionally save them to `~/.config/asklake/credentials.json` (`0600`). Credentials entered there are sent per request and never stored on the server.

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
