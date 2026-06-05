"""Real headline eval: run baseline / agentic / semantic over the real IMDb gold set against a
LIVE LLM and print the execution-accuracy comparison. This is the resume-grade run behind the
illustrative hermetic `make eval` numbers.

Run it (manual; needs a key + the built IMDb parquet — `make build-imdb`):
    DEEPSEEK_API_KEY=...  make eval-real     # uses DeepSeekProvider (deepseek-chat)
    ANTHROPIC_API_KEY=... make eval-real     # falls back to AnthropicProvider

`run_real_eval` is provider/backend-agnostic (hermetically tested with FakeLLMProvider + an
in-memory backend); only the CLI wires the live provider + the IMDb parquet backend."""

from __future__ import annotations

import os

from engine.lakehouse.duckdb_backend import DuckDBBackend
from engine.ports.llm import LLMProvider
from engine.ports.storage import StorageBackend
from engine.semantic.semantic_model import SemanticLayer, load_semantic_layer
from eval.harness import EvalCase, SystemReport, score_case
from eval.imdb_gold import IMDB_GOLD, PARQUET_DIR
from eval.systems import run_agentic, run_baseline, run_semantic

_SEMANTIC_YAML = "datasets/imdb_cmu/semantic.yaml"


def apply_duckdb_guardrails(
    backend, memory_limit: str = "2GB", max_temp_size: str = "4GB", threads: int = 2
) -> None:
    """Cap the eval backend's RAM + on-disk spill so a pathological candidate query fails fast
    (catchable error) instead of exhausting the machine. Each SET is best-effort (older DuckDB
    builds may not support every knob)."""
    for stmt in (
        f"SET memory_limit='{memory_limit}'",
        f"SET max_temp_directory_size='{max_temp_size}'",
        f"SET threads={threads}",
    ):
        try:
            backend.run_sql(stmt)
        except Exception:  # noqa: BLE001
            pass


def run_real_eval(
    llm: LLMProvider,
    backend: StorageBackend,
    cases: list[EvalCase],
    layer: SemanticLayer,
    max_retries: int = 2,
) -> list[SystemReport]:
    """Score the three systems over `cases` against a shared (read-only) backend. Candidate and
    gold SQL are both SELECTs executed on `backend` via score_case."""
    runners = {
        "baseline": lambda c: run_baseline(llm, backend, c.question),
        "agentic": lambda c: run_agentic(llm, backend, c.question, max_retries),
        "semantic": lambda c: run_semantic(llm, backend, c.question, layer, max_retries),
    }
    reports: list[SystemReport] = []
    for name, runner in runners.items():
        valid = correct = 0
        attempts_total = 0
        for case in cases:
            sql = None
            attempts = 0
            v = ok = False
            for _attempt in range(2):  # 1 try + 1 retry
                try:
                    sql, attempts = runner(case)
                    v, ok = score_case(sql, case.gold_sql, backend)
                    break
                except Exception:  # noqa: BLE001
                    sql, attempts, v, ok = None, 0, False, False
            valid += int(v)
            correct += int(ok)
            attempts_total += attempts
        n = len(cases) or 1
        reports.append(
            SystemReport(
                name=name,
                n=len(cases),
                valid_sql_rate=valid / n,
                execution_accuracy=correct / n,
                avg_attempts=attempts_total / n,
            )
        )
    return reports


def _make_provider() -> LLMProvider:
    if os.environ.get("DEEPSEEK_API_KEY"):
        from engine.llm.deepseek_provider import DeepSeekProvider

        return DeepSeekProvider(
            model=os.environ.get("ASKLAKE_DEEPSEEK_MODEL", "deepseek-chat"), timeout=120.0
        )
    if os.environ.get("ANTHROPIC_API_KEY"):
        from engine.llm.anthropic_provider import AnthropicProvider
        from engine.settings import get_settings

        return AnthropicProvider(model=get_settings().llm_model)
    raise SystemExit(
        "No API key found. Set DEEPSEEK_API_KEY (or ANTHROPIC_API_KEY) to run the real eval."
    )


def main() -> None:
    from pathlib import Path

    if not Path(PARQUET_DIR).exists():
        raise SystemExit(
            f"IMDb parquet not found at {PARQUET_DIR}. Build it first: make build-imdb"
        )
    backend = DuckDBBackend(parquet_dir=PARQUET_DIR)
    apply_duckdb_guardrails(backend)
    layer = load_semantic_layer(_SEMANTIC_YAML)
    llm = _make_provider()
    reports = run_real_eval(llm, backend, IMDB_GOLD, layer)
    print(f"Real IMDb eval ({len(IMDB_GOLD)} cases), provider={type(llm).__name__}")
    print(f"{'system':<12}{'n':>4}{'valid%':>9}{'exec-acc':>10}{'avg-retries':>13}")
    for r in reports:
        print(
            f"{r.name:<12}{r.n:>4}{r.valid_sql_rate:>8.0%}"
            f"{r.execution_accuracy:>10.0%}{r.avg_attempts:>13.2f}"
        )
    print("eval-real OK")


if __name__ == "__main__":
    main()
