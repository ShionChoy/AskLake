"""`make eval` entrypoint: prints the offline baseline-vs-agentic comparison table.
Headline numbers come from a real-LLM run over a BIRD/Spider subset (see docs/eval.md)."""

from __future__ import annotations

from eval.hermetic import run_hermetic_comparison
from eval.semantic_hermetic import run_semantic_comparison


def main() -> None:
    baseline, agentic = run_hermetic_comparison()
    print(f"{'system':<26}{'n':>4}{'valid%':>9}{'exec-acc':>10}{'avg-retries':>13}")
    for r in (baseline, agentic):
        print(
            f"{r.name:<26}{r.n:>4}{r.valid_sql_rate:>8.0%}"
            f"{r.execution_accuracy:>10.0%}{r.avg_attempts:>13.2f}"
        )
    print()
    raw, semantic = run_semantic_comparison()
    print(f"{'system':<26}{'n':>4}{'valid%':>9}{'exec-acc':>10}{'avg-retries':>13}")
    for r in (raw, semantic):
        print(
            f"{r.name:<26}{r.n:>4}{r.valid_sql_rate:>8.0%}"
            f"{r.execution_accuracy:>10.0%}{r.avg_attempts:>13.2f}"
        )
    print("eval OK")


if __name__ == "__main__":
    main()
