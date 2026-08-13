# Running AskLake in the browser

A step-by-step guide to launch AskLake on **real IMDb data** with a live LLM (DeepSeek by
default) and use it from your browser, plus how to **stop it cleanly**.

This guide assumes the `uv` toolchain. Run every command from the project root.

---

## What actually runs

Two local processes talk to each other over HTTP:

| Process | Command | Port | Role |
|---|---|---|---|
| **API** | `make serve` → `uvicorn api.serve:build_app --factory` | `8000` | NL→SQL `/ask_trace` + `/ask`, raw `/query`, `/info`, `/metrics` |
| **UI** | `make ui` → `streamlit run ui/app.py` | `8501` | the web page you open in the browser |

The API serves the **grounded + self-correcting (agentic)** SQL path: an LLM + the semantic
layer + a **2 GB DuckDB memory cap** (so a runaway query fails fast instead of OOM-killing WSL).
The UI shows the selected model and a live trace of the backend steps.

**You can supply the LLM key three ways** (pick one):

1. **In the browser** — boot the API with *no* key and paste your key in the **⚙️ Model & API
   key** sidebar (this is the bring-your-own-key flow; see §4). The key is sent per request and
   never stored on the server.
2. **`.env` file** — `cp .env.example .env`, set `DEEPSEEK_API_KEY=sk-...`. `make serve` loads it
   automatically.
3. **Inline env var** — `DEEPSEEK_API_KEY=sk-... make serve` (env only — never commit it).

---

## 0. One-time prerequisites

```bash
cd ~/projects/application
uv sync          # installs deps incl. httpx (needed by the DeepSeek provider)
```

You also need a **DeepSeek API key** (`sk-...`) — or an Anthropic key if you prefer Claude. The
key is optional at boot; without one the API still serves `/query` and `/health`, and you can
enter the key in the browser.

> **Memory note (WSL):** the WSL guest sees only ~6–8 GB of the 16 GB host. The build below is
> capped at 4 GB and the running server at 2 GB, so neither should OOM the guest.

---

## 1. One-time data setup (real IMDb → Parquet)

### 1a. Download the raw IMDb data

```bash
bash scripts/download_data.sh
```

This pulls the IMDb TSVs (incl. **`title.principals.tsv.gz` ≈ 768 MB** — required for cast /
"who starred in" questions) into `data/` (gitignored, never committed).

> If you already have a **partial** `data/imdb/raw/` and only need the cast table, fetch just it:
> ```bash
> curl -fL -o data/imdb/raw/title.principals.tsv.gz https://datasets.imdbws.com/title.principals.tsv.gz
> ```

### 1b. Build the Parquet working set

```bash
make build-imdb            # ~1–2 min; smaller/faster set: MIN_VOTES=5000 make build-imdb
```

Verify you get **5** Parquet files:

```bash
ls -la data/imdb/parquet/
# name_basics.parquet  title_basics.parquet  title_crew.parquet  title_principals.parquet  title_ratings.parquet
```

These register as DuckDB **views**: `title_basics`, `title_ratings`, `title_crew`,
`name_basics`, `title_principals` — the schema the LLM (and you) query.

### 1c. (Optional) Build the knowledge graph

GraphRAG (graph / fusion paths) needs a one-time graph build. The structured backbone (genres,
year, directors, cast, characters) comes straight from the IMDb parquet; plot **themes** are
LLM-extracted from **current English Wikipedia** plot text (resolved via a Wikidata P345 IMDb-ID
bridge), so the build needs network access + an LLM key:

```bash
GRAPH_FILMS=200 DEEPSEEK_API_KEY=sk-... make build-graph
# -> data/imdb/graph/triples.jsonl  (the API auto-loads it at boot)
```

`GRAPH_FILMS` caps how many top-voted films get Wikipedia-theme extraction (one LLM call each);
the structured triples cover all films ≥1000 votes regardless. Start small (`GRAPH_FILMS=30`) to
try it cheaply. Without a built graph, the browser still answers SQL questions; graph/fusion just
isn't available. (This same `triples.jsonl` is what the optional Neo4j backend loads — see
*Optional: GraphRAG on Neo4j*.)

---

## 2. Launch (two terminals)

Open **two** terminals, both at `~/projects/application`.

### Terminal 1 — the API

Simplest (with `/metrics` enabled). Pick a key source from "What actually runs" above; this
example boots **keyless** so you can supply the key in the browser:

```bash
ASKLAKE_OBSERVABILITY_BACKEND=prometheus make serve
```

`make serve` runs `uvicorn api.serve:build_app --factory --host 0.0.0.0 --port 8000`, adding
`--env-file .env` automatically when a `.env` exists.

- **Keyless boot** prints: `[api.serve] no default LLM provider (...); supply a key in the UI sidebar`
  — this is expected; the app is up, just waiting for a key.
- **With a key** (via `.env` or inline), uvicorn simply prints `Uvicorn running on http://0.0.0.0:8000`.

Equivalent explicit command (e.g. to boot with an inline key):

```bash
DEEPSEEK_API_KEY=sk-... ASKLAKE_OBSERVABILITY_BACKEND=prometheus \
  uv run uvicorn api.serve:build_app --factory --host 0.0.0.0 --port 8000
```

### Terminal 2 — the Streamlit UI

```bash
make ui
```

Equivalent explicit command:

```bash
uv run streamlit run ui/app.py \
  --server.headless true --server.port 8501 --server.address 0.0.0.0 \
  --browser.gatherUsageStats false
```

`--server.headless true` skips the first-run email prompt and the auto-open attempt
(which doesn't work cleanly under WSL anyway).

### Open the browser

Go to **http://localhost:8501** in your Windows browser — WSL2 forwards `localhost`, so the
plain URL works.

---

## 3. Confirm it's healthy (optional)

From a third terminal:

```bash
curl -s localhost:8000/health                       # {"status":"ok",...}
curl -s localhost:8000/info | uv run python -m json.tool
# keyless boot:  {"provider":"(client-supplied)","model":"(set in the sidebar)", ...}
# with a key:    {"provider":"DeepSeekProvider","model":"deepseek-v4-flash", ...}
```

---

## 4. Using it in the browser

### ⚙️ Model & API key (the sidebar)

The left sidebar is where you bring your own key:

- **Provider** — `deepseek` or `anthropic`.
- **Model** — pick from the provider's defaults (e.g. `deepseek-v4-flash`, `deepseek-v4-pro`) or
  choose **(custom…)** to type any model name.
- **API key** — paste your key (the field is masked).
- **Save locally** — writes `~/.config/asklake/credentials.json` (mode `0600`, on your machine
  only). Next launch, the sidebar pre-fills from it.
- **Delete saved key** — removes that file and clears the field.

The key travels in each request body and is **never persisted on the server**, never logged, and
never echoed back (even error messages have the key redacted). If you click **Ask** with no key
entered, you get a friendly *"Enter your API key in the sidebar to ask questions"* message
instead of an error.

### Retrieval path

The sidebar's **Retrieval path** selector chooses how a question is answered:
- **Auto** — the Router scores the question and picks SQL, graph, or fuses both.
- **SQL** — force the Text-to-SQL path.
- **Graph** — force GraphRAG (multi-hop over the knowledge graph; **needs no API key**).
- **Fusion** — run SQL and graph and merge (SQL table + cited graph narrative).

Graph/fusion answers show an info box with the narrative; each fact is cited to its source plot.

### Asking questions

Under the title you'll see the active model, e.g. `🧠 model: deepseek-v4-flash · provider: deepseek`.

- **Ask in natural language** — type a question, click **Ask**. You'll see a
  **Backend processing steps** timeline (schema retrieval → SQL generation → execution, with
  ✅/❌ status, timings, and the actual SQL). A red ❌ "Execute SQL" step followed by another
  "Generate SQL" is the **self-correction** loop working. Then the final SQL, table, and chart.
- **Raw SQL console** — type SQL directly (works even without a key).

Good test questions:
- `Top 10 highest-rated movies since 2015 with at least 100000 votes`
- `The top ten highest-rated movies Keanu Reeves has starred in`
- `Which directors have the most movies rated above 8.0?`

> First NL answer can take ~10–60 s (DeepSeek latency); the UI waits up to 180 s.

---

## 5. Stopping it properly

### If you launched in the foreground (the setup above)

Press **`Ctrl-C`** in **each** terminal (Terminal 2 for Streamlit, Terminal 1 for the API).
That's the clean stop.

### If they're running in the background, or Ctrl-C didn't catch them

Stop both by process pattern, then **verify the ports are free**:

```bash
pkill -f "uvicorn api.serve"          # stop the API
pkill -f "streamlit run ui/app.py"    # stop the UI

# verify nothing is still listening (no output = fully stopped):
ss -ltn | grep -E ':8000|:8501' || echo "stopped — ports 8000/8501 are free"
```

Streamlit sometimes survives the first `pkill` under WSL (a grandchild process keeps the port).
If `:8501` is still held, kill it directly:

```bash
fuser -k 8501/tcp        # or: kill -9 $(pgrep -f "streamlit run ui/app.py")
```

### Cleanup notes

- **Nothing to clean up.** DuckDB runs in-memory over the Parquet **views**; stopping the
  server discards that. Your built data stays in `data/` (gitignored).
- A saved key lives at `~/.config/asklake/credentials.json` until you click **Delete saved key**
  (or `rm` it). It's outside the repo and gitignored by the secrets rules anyway.

---

## Optional: GraphRAG on Neo4j

By default the knowledge graph is **in-process**, loaded from `triples.jsonl` at boot with no
extra service. You can instead connect AskLake to an independently managed **Neo4j** server, where
the graph lives in the database and typed retrieval executes as Cypher. The in-process default is
unchanged and is what CI uses.

**Prerequisites:** a reachable Neo4j server, the optional Python driver, and a built graph
(§1c → `data/imdb/graph/triples.jsonl`).

```bash
uv sync --extra neo4j
NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=... \
  make graph-load-neo4j
```

Then launch the API **pointed at Neo4j** (Terminal 1) and the UI as usual (Terminal 2):

```bash
# Terminal 1 — API on the Neo4j backend:
ASKLAKE_GRAPH_BACKEND=neo4j NEO4J_URI=bolt://localhost:7687 \
  NEO4J_USER=neo4j NEO4J_PASSWORD=... make serve
# Terminal 2 — UI:
make ui
```

> `NEO4J_PASSWORD` is required and must match the server. Set it inline as above or put the
> connection values in `.env`. If AskLake cannot authenticate or connect, it logs the reason and
> falls back to the in-memory graph so the API can still boot.

**Confirm it really connected** — the API boot log should print:

```
[api.serve] graph backend: neo4j (bolt://localhost:7687)
```

(If you instead see `neo4j unavailable (...); falling back to in-memory graph`, Neo4j isn't
reachable or the password is wrong.)

**Use it:** in the browser pick the **Graph** retrieval path (needs no API key) and ask e.g.
*"who acted in The Dark Knight"* or *"what themes do Inception and Interstellar share"* — those
answers traverse Neo4j via Cypher. The grounded natural-language graph answer still needs an LLM
key; without one you get the cited triples + the 🕸️ Network view.

If your Neo4j installation includes Browser, you can inspect the graph directly:

```cypher
MATCH (f:Film {name:'The Dark Knight'})-[r]-(n) RETURN f, r, n LIMIT 50
```

Neo4j lifecycle, persistence, backup, and shutdown are managed outside this repository.

---

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Ask returns "Enter your API key in the sidebar" | No usable key. Paste one in the **⚙️ Model & API key** sidebar, or boot the API with a key (`.env` / inline). |
| Ask returns "The model call failed: … Check your API key and model" | The key or model is wrong/expired (e.g. a 401). Fix the key, or pick a valid model. The key is redacted from this message. |
| `IMDb parquet not found` / empty results | Run §1: `bash scripts/download_data.sh` then `make build-imdb`. |
| Cast / "starred in" questions return too few rows | You need `title_principals`. Ensure `data/imdb/parquet/title_principals.parquet` exists; if not, fetch `title.principals.tsv.gz` (§1a) and re-run `make build-imdb`. |
| WSL gets killed during `make build-imdb` | Use `MIN_VOTES=5000 make build-imdb`, or raise WSL RAM: add `memory=12GB` under `[wsl2]` in `%UserProfile%\.wslconfig` (Windows), then `wsl --shutdown`. The build is capped at 4 GB so it should spill to disk rather than OOM. |
| Browser page shows a connection error | The API (Terminal 1) isn't up yet, or is on a different port. Check `curl localhost:8000/health`. |
| `Address already in use` on launch | A previous server is still running — stop it (see §5), confirm with `ss -ltn \| grep -E ':8000\|:8501'`. |
| `/ask_trace` is slow or the UI spins | DeepSeek latency (~10–60 s); a self-correction adds a round. UI timeout is 180 s. |
| Graph answers empty after switching to Neo4j | If the boot log shows `falling back to in-memory`, verify the server is reachable and all three `NEO4J_*` values are correct. A successful connection logs `graph backend: neo4j (bolt://localhost:7687)`. |
| Want different ports | API: `ASKLAKE_API_PORT=8001 make serve`. UI: `ASKLAKE_API_URL=http://localhost:8001 uv run streamlit run ui/app.py --server.port 8502 …` (point the UI at the API's URL). |

---

## Appendix A — API endpoints

| Method | Path | Returns |
|---|---|---|
| GET | `/health` | `{"status":"ok"}` |
| GET | `/info` | `{provider, model, path}` (default provider, or `(client-supplied)` when keyless) |
| POST | `/query` | `{columns, rows}` for a raw SQL body `{"sql": "..."}` |
| POST | `/ask` | NL→SQL: `{path, sql, columns, rows, chart_spec, narrative}` (credential-less fallback) |
| POST | `/ask_trace` | same as `/ask` **plus** `{model, steps[], elapsed_ms}`; accepts optional `provider`/`model`/`api_key` in the body (the UI prefers this) |
| GET | `/metrics` | Prometheus exposition (active because `ASKLAKE_OBSERVABILITY_BACKEND=prometheus`) |

## Appendix B — environment variables

| Variable | Used by | Purpose |
|---|---|---|
| `DEEPSEEK_API_KEY` | API | DeepSeek auth at boot (optional — can be supplied in the browser instead) |
| `ANTHROPIC_API_KEY` + `ASKLAKE_LLM_PROVIDER=anthropic` | API | use Claude instead of DeepSeek |
| `ASKLAKE_OBSERVABILITY_BACKEND` | API | `prometheus` enables the `/metrics` endpoint (default `noop`) |
| `ASKLAKE_PARQUET_DIR` | API | Parquet location (default `data/imdb/parquet`) |
| `ASKLAKE_API_HOST` / `ASKLAKE_API_PORT` | API | bind host/port (default `0.0.0.0:8000`) |
| `ASKLAKE_API_URL` | UI | which API the UI calls (default `http://localhost:8000`) |
| `MIN_VOTES` | `make build-imdb` | min votes to include a movie (default `1000`) |
| `ASKLAKE_GRAPH_BACKEND` | API | `memory` (default, in-process) or `neo4j` (see *Optional: GraphRAG on Neo4j*) |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | API + loader | Connection settings for an externally managed Neo4j server when backend=`neo4j` |

> The browser sidebar's saved credentials live in `~/.config/asklake/credentials.json` (`0600`),
> managed entirely by the UI — the API never reads or writes it.

---

## Appendix C — quick reference (copy-paste)

```bash
# ── start ────────────────────────────────────────────────
cd ~/projects/application
# terminal 1 (API) — keyless; paste your key in the browser sidebar:
ASKLAKE_OBSERVABILITY_BACKEND=prometheus make serve
#   (or boot with a key: DEEPSEEK_API_KEY=sk-... make serve)
# terminal 2 (UI):
make ui
# then open http://localhost:8501

# ── stop ─────────────────────────────────────────────────
# Ctrl-C in each terminal, OR:
pkill -f "uvicorn api.serve"; pkill -f "streamlit run ui/app.py"
ss -ltn | grep -E ':8000|:8501' || echo "stopped"
```
