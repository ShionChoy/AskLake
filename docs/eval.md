# AskLake evaluation

AskLake uses execution-based evaluation: candidate SQL and gold SQL run against the same DuckDB
backend, and their rows are compared as order-insensitive multisets. Whole numbers and text match
exactly; non-integral numeric values use a small relative tolerance.

## Metrics

- **Execution accuracy**: the fraction of candidates whose result matches the gold result.
- **Valid-SQL rate**: the fraction of candidates that execute successfully.
- **Average corrections**: self-correction rounds used per question.
- **LLM calls/question and wall time/question**: cost columns for live ablations.
- **Per-tier accuracy**: aggregation, top-N, and multi-hop breakdowns.

The comparison and scoring code lives in `eval/harness.py`, while `eval/systems.py` assembles the
systems under comparison.

## Hermetic evaluation

```bash
make eval
```

This deterministic run uses `FakeLLMProvider` and needs no API key or downloaded data. It checks:

- single-prompt SQL versus self-correcting SQL;
- raw schema versus semantic-layer grounding;
- SQL/graph/fusion routing decisions;
- whether graph results carry source citations.

The small canned cases validate the harness and wiring; they are not benchmark claims. CI runs
this command after the test suite.

## Live ablation

IMDb:

```bash
bash scripts/download_data.sh
make build-imdb MIN_VOTES=25
DEEPSEEK_API_KEY=... make eval-real
```

Synthetic CRM:

```bash
make build-crm
DEEPSEEK_API_KEY=... make eval-real-crm
```

Anthropic can be used by setting `ANTHROPIC_API_KEY` and the corresponding provider settings.
The live runner compares five cumulative systems:

1. raw-schema, single-prompt baseline;
2. semantic grounding plus self-correction;
3. value linking;
4. planning plus K-candidate self-consistency;
5. the full grounded path with a result critic.

IMDb cases are defined in `eval/imdb_gold.py`; CRM cases are in `eval/crm_gold.py`. Dataset
selection only changes the Parquet path, semantic configuration, and gold set. The engine and
scorer remain the same.

## Interpreting results

The repository README records the latest measured tables and caveats. In particular, the CRM set
is intentionally small and its per-tier percentages have high variance. Treat the live results as
a reproducible engineering comparison, not a statistically stable model leaderboard.
