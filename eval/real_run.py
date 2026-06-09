"""Real headline eval: run baseline / agentic / semantic over the real IMDb gold set against a
LIVE LLM and print the execution-accuracy comparison. This is the resume-grade run behind the
illustrative hermetic `make eval` numbers.

Run it (manual; needs a key + the built IMDb parquet — `make build-imdb`):
    DEEPSEEK_API_KEY=...  make eval-real     # uses DeepSeekProvider (deepseek-v4-flash)
    ANTHROPIC_API_KEY=... make eval-real     # falls back to AnthropicProvider

`run_real_eval` is provider/backend-agnostic (hermetically tested with FakeLLMProvider + an
in-memory backend); only the CLI wires the live provider + the IMDb parquet backend."""

from __future__ import annotations

import os
import time

from engine.lakehouse.duckdb_backend import DuckDBBackend
from engine.ports.llm import LLMProvider
from engine.ports.storage import StorageBackend
from engine.semantic.semantic_model import SemanticLayer, load_semantic_layer
from eval._counting import CountingLLM
from eval.harness import EvalCase, SystemReport, score_case
from eval.systems import (
    run_agentic,
    run_baseline,
    run_grounded,
    run_plan_sc,
    run_semantic,
    run_value_link,
)

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
        tier_correct: dict[str, int] = {}
        tier_total: dict[str, int] = {}
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
            if case.tier:
                tier_total[case.tier] = tier_total.get(case.tier, 0) + 1
                tier_correct[case.tier] = tier_correct.get(case.tier, 0) + int(ok)
        n = len(cases) or 1
        per_tier = (
            {t: tier_correct[t] / tier_total[t] for t in sorted(tier_total)} if tier_total else None
        )
        reports.append(
            SystemReport(
                name=name,
                n=len(cases),
                valid_sql_rate=valid / n,
                execution_accuracy=correct / n,
                avg_attempts=attempts_total / n,
                per_tier=per_tier,
            )
        )
    return reports


def run_ablation_eval(
    llm: LLMProvider,
    backend: StorageBackend,
    cases: list[EvalCase],
    layer: SemanticLayer,
    value_index=None,
    k_candidates: int = 3,
    max_retries: int = 2,
) -> list[SystemReport]:
    """Five-rung ablation (baseline -> +semantic -> +value-link -> +plan/sc -> grounded), per tier,
    with avg LLM-calls + wall-ms cost columns. Each case is wrapped in a CountingLLM and timed; a
    failing case (1 retry) is scored 0 rather than aborting the run (matches run_real_eval)."""
    runners = {
        "baseline": lambda c, m: run_baseline(m, backend, c.question),
        "+semantic": lambda c, m: run_semantic(m, backend, c.question, layer, max_retries),
        "+value-link": lambda c, m: run_value_link(
            m, backend, c.question, layer, value_index, max_retries
        ),
        "+plan/sc": lambda c, m: run_plan_sc(
            m, backend, c.question, layer, value_index, k_candidates, max_retries
        ),
        "grounded": lambda c, m: run_grounded(
            m, backend, c.question, layer, value_index, k_candidates, max_retries
        ),
    }
    reports: list[SystemReport] = []
    for name, runner in runners.items():
        valid = correct = attempts_total = calls_total = 0
        wall_total = 0.0
        tier_correct: dict[str, int] = {}
        tier_total: dict[str, int] = {}
        for case in cases:
            counting = CountingLLM(llm)
            v = ok = False
            attempts = 0
            t0 = time.perf_counter()
            for _attempt in range(2):  # 1 try + 1 retry (transient API robustness)
                try:
                    sql, attempts = runner(case, counting)
                    v, ok = score_case(sql, case.gold_sql, backend)
                    break
                except Exception:  # noqa: BLE001
                    sql, attempts, v, ok = None, 0, False, False
            wall_total += (time.perf_counter() - t0) * 1000
            calls_total += counting.calls
            valid += int(v)
            correct += int(ok)
            attempts_total += attempts
            if case.tier:
                tier_total[case.tier] = tier_total.get(case.tier, 0) + 1
                tier_correct[case.tier] = tier_correct.get(case.tier, 0) + int(ok)
        n = len(cases) or 1
        per_tier = (
            {t: tier_correct[t] / tier_total[t] for t in sorted(tier_total)} if tier_total else None
        )
        reports.append(
            SystemReport(
                name=name,
                n=len(cases),
                valid_sql_rate=valid / n,
                execution_accuracy=correct / n,
                avg_attempts=attempts_total / n,
                per_tier=per_tier,
                avg_llm_calls=calls_total / n,
                avg_wall_ms=wall_total / n,
            )
        )
    return reports


def eval_dataset_config(name: str) -> dict:
    """Resolve per-dataset eval wiring. Keeps real_run dataset-agnostic (selection is config)."""
    if name == "crm":
        from eval.crm_gold import CRM_GOLD, PARQUET_DIR

        return {
            "parquet_dir": PARQUET_DIR,
            "semantic_yaml": "datasets/crm_demo/semantic.yaml",
            "gold": lambda: CRM_GOLD,
        }
    from eval.imdb_gold import IMDB_GOLD
    from eval.imdb_gold import PARQUET_DIR as IMDB_PARQUET

    return {
        "parquet_dir": IMDB_PARQUET,
        "semantic_yaml": _SEMANTIC_YAML,
        "gold": lambda: IMDB_GOLD,
    }


def _make_provider() -> LLMProvider:
    if os.environ.get("DEEPSEEK_API_KEY"):
        from engine.llm.deepseek_provider import DeepSeekProvider

        return DeepSeekProvider(
            model=os.environ.get("ASKLAKE_DEEPSEEK_MODEL", "deepseek-v4-flash"), timeout=120.0
        )
    if os.environ.get("ANTHROPIC_API_KEY"):
        from engine.llm.anthropic_provider import AnthropicProvider
        from engine.settings import get_settings

        return AnthropicProvider(model=get_settings().llm_model)
    raise SystemExit(
        "No API key found. Set DEEPSEEK_API_KEY (or ANTHROPIC_API_KEY) to run the real eval."
    )


def main() -> None:
    import sys
    from pathlib import Path

    from engine.semantic.value_index import build_value_index

    name = os.environ.get("ASKLAKE_EVAL_DATASET", "imdb")
    if "--dataset" in sys.argv:
        name = sys.argv[sys.argv.index("--dataset") + 1]
    cfg = eval_dataset_config(name)
    if not Path(cfg["parquet_dir"]).exists():
        raise SystemExit(
            f"{name} parquet not found at {cfg['parquet_dir']}. Build it first "
            f"(make build-{'imdb MIN_VOTES=25' if name == 'imdb' else 'crm'})."
        )
    backend = DuckDBBackend(parquet_dir=cfg["parquet_dir"])
    apply_duckdb_guardrails(backend)
    layer = load_semantic_layer(cfg["semantic_yaml"])
    value_index = build_value_index(layer, backend)
    cases = cfg["gold"]()
    llm = _make_provider()
    reports = run_ablation_eval(llm, backend, cases, layer, value_index=value_index)

    print(f"Ablation eval — dataset={name}, {len(cases)} cases, provider={type(llm).__name__}")
    print(f"{'system':<14}{'n':>4}{'valid%':>9}{'exec-acc':>10}{'llm/q':>8}{'ms/q':>9}")
    for r in reports:
        print(
            f"{r.name:<14}{r.n:>4}{r.valid_sql_rate:>8.0%}{r.execution_accuracy:>10.0%}"
            f"{r.avg_llm_calls:>8.1f}{r.avg_wall_ms:>9.0f}"
        )
    tiers = sorted({c.tier for c in cases if c.tier})
    if tiers:
        print(f"\n{'by-tier exec-acc':<16}" + "".join(f"{t:>14}" for t in tiers))
        for r in reports:
            pt = r.per_tier or {}
            print(f"{r.name:<16}" + "".join(f"{pt.get(t, 0.0):>13.0%} " for t in tiers))
    print("eval-real OK")


if __name__ == "__main__":
    main()
