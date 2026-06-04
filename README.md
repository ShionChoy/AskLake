# AskLake

Governed multi-agent natural-language analytics over your own data
(Text-to-SQL + GraphRAG). See architecture docs (local).

## Quickstart (dev)

```bash
# 1. Install deps
uv sync

# 2. Run the P0 in-process smoke demo (no Docker, no API key required)
make demo-p0
# Expected output: columns + 2 rows (Interstellar, Tenet) + "demo-p0 OK"

# 3. Copy the env template (edit values as needed)
cp .env.example .env

# 4. Start the full core stack (requires Docker)
make dev       # docker compose --profile core up
# Services: FastAPI app on :8000, Postgres on :5432, Qdrant on :6333

# 5. Download raw datasets into ./data (gitignored)
bash scripts/download_data.sh

# 6. Lint + test
make lint
make test
```

## Phase 1: NL→SQL

Ask natural-language questions over IMDb. The `/ask` endpoint accepts a question, generates SQL via the LangGraph agent, runs it against the Parquet-backed lakehouse, and returns a table + bar-chart spec.

**Example question:** "Highest-rated sci-fi films after 2010 (top 10)"

### Build the IMDb working set

```bash
# Download raw TSVs into ./data (gitignored — never committed)
./scripts/download_data.sh

# Convert TSVs → Parquet (MIN_VOTES default 1000, configurable)
make build-imdb
# Writes parquet files to data/; safe to re-run.
```

### Run the hermetic demo (no API key, no data download required)

```bash
make demo-p1
# Expected: 2 stub rows (Sci A 8.9, Sci B 7.5) + chart spec + "demo-p1 OK"
```

### Run the real interactive stack

Set `ANTHROPIC_API_KEY` in your environment (or in `.env` — the provider reads
`ANTHROPIC_API_KEY` directly; no `ASKLAKE_` prefix required). Point
`PARQUET_DIR` at the built parquet directory, then:

```bash
make dev   # docker compose --profile core up → FastAPI on :8000
```

**Data licenses:** IMDb is non-commercial (no redistribution); CMU is CC BY-SA.

## Phase 2: Agentic self-correction + evaluation

The NL→SQL path is now **agentic**: a cyclic LangGraph (`SQLWriter → Validator → SelfCorrect ≤N → … → END`) executes the generated SQL, and on failure feeds the error back to the model for a bounded number of correction attempts. `AgenticSqlPath` is an additive sibling of the Phase 1 `SqlPath` (the one-shot graph is kept as the evaluation baseline).

**Example self-correction:** the model first writes `SELECT title, rating FROM movies …` (no such column); the validator's error (`Referenced column "rating" not found`) is fed back, and the corrected `SELECT title, averageRating FROM movies …` succeeds.

### Run the hermetic Phase 2 demo (no API key)

```bash
make demo-p2
# Shows one live self-correction (rating → averageRating) + the eval comparison table.
```

### Evaluation

We quantify the agentic lift with an **Execution-Accuracy** harness (result-set multiset match vs a gold SQL):

- **Execution Accuracy** — fraction of questions whose result set matches the gold query's;
- **Valid-SQL Rate** — fraction whose SQL executes without error;
- **avg self-corrections** — mean correction rounds per question.

```bash
make eval   # hermetic baseline-vs-agentic comparison, no API key
```

The committed `make eval` runs on a tiny **illustrative** hermetic fixture (3-case toy set) that demonstrates the harness and the self-correction mechanism: the single-prompt baseline scores 67% execution accuracy, the agentic self-correct loop scores 100%.

> **Headline numbers** (baseline vs agentic over a real BIRD/Spider subset, run with a live LLM) — `TODO: paste real benchmark table`. Methodology + reproduction recipe live in `docs/eval.md` (local).

## Phase 3: Semantic layer + governance

The SQL path is now **grounded**: a `SemanticLayerProvider` supplies the LLM with curated table/column descriptions, metrics, synonyms (e.g. "score" → `averageRating`), and few-shot SQL examples, pruned to the question by a pluggable `SchemaRetriever` (in-process lexical now; Qdrant-backed later). It is an additive sibling of the bare `RawSchemaProvider` (kept as the eval baseline). Dataset semantics live in `datasets/imdb_cmu/semantic.yaml`.

**Governance** — `PolicyGovernance` enforces `datasets/imdb_cmu/governance.yaml`:

- **RBAC + PII masking** — e.g. `birthYear`/`deathYear` masked for the `public` role
- **Row-level filtering** — e.g. `adult` titles hidden from `public`
- **Query cost guardrail** — `before_query` blocks unbounded scans (no LIMIT) and non-SELECT writes, raising `GovernanceError`

### Run the hermetic Phase 3 demo (no API key)

```bash
make demo-p3
# Same question under `analyst` vs `public`: full vs masked+filtered rows,
# then one cost-guardrail interception.
```

### Evaluation

`make eval` now prints a **second comparison table** — bare schema vs semantic layer — illustrating the grounding lift (toy: 0% → 100% on a synonym case where the raw schema keeps emitting a non-existent `score` column). Same illustrative / real-numbers-TODO framing as Phase 2; headline numbers come from a manual real-LLM run.

> **Headline numbers** (raw-schema agentic vs semantic-layer agentic over a real BIRD/Spider subset) — `TODO: paste real benchmark table`. Methodology + reproduction recipe live in `docs/eval.md` (local).

## License
Apache-2.0. IMDb data is non-commercial (not redistributed); CMU corpus is CC BY-SA.
