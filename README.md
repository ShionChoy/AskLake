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

## License
Apache-2.0. IMDb data is non-commercial (not redistributed); CMU corpus is CC BY-SA.
