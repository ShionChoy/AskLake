"""Print the deterministic offline SQL, semantic-grounding, and routing comparisons."""

from __future__ import annotations

from eval.hermetic import run_hermetic_comparison
from eval.routing_hermetic import run_graph_grounding, run_routing_eval
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
    print()
    rep = run_routing_eval()
    print(f"{'router decision':<58}{'expected':>10}{'actual':>10}{'ok':>5}")
    for question, expected, actual, ok in rep.rows:
        print(f"{question[:56]:<58}{expected:>10}{actual:>10}{('ok' if ok else 'NO'):>5}")
    print(f"routing accuracy: {rep.accuracy:.0%} ({rep.n} cases)")
    no_graph, grounded = run_graph_grounding()
    print(f"graph-grounding (illustrative): no-graph {no_graph}/1 vs graph {grounded}/1 cited")
    print("eval OK")


if __name__ == "__main__":
    main()
