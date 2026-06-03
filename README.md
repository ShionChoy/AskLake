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

## License
Apache-2.0. IMDb data is non-commercial (not redistributed); CMU corpus is CC BY-SA.
