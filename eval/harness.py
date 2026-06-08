from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from engine.lakehouse.duckdb_backend import DuckDBBackend
from engine.ports.storage import StorageBackend


@dataclass(frozen=True)
class EvalCase:
    name: str
    schema_sql: str  # DDL seeding the case database (DuckDB, in-memory)
    question: str
    gold_sql: str
    tier: str = ""  # difficulty/family label: "aggregation" | "topn" | "multihop" (eval only)


@dataclass(frozen=True)
class SystemReport:
    name: str
    n: int
    valid_sql_rate: float
    execution_accuracy: float
    avg_attempts: float
    per_tier: dict[str, float] | None = None  # exec-acc per tier (None when no tiers present)


def _rows_match(a, b) -> bool:
    # Execution Accuracy compares result sets as multisets (order-insensitive).
    return Counter(map(tuple, a)) == Counter(map(tuple, b))


def score_case(candidate_sql: str, gold_sql: str, backend: StorageBackend) -> tuple[bool, bool]:
    """Return (valid, correct): valid = candidate executes; correct = result multiset == gold's.

    A gold query that fails to execute is a malformed EvalCase (not a system-under-test result),
    so it raises with the offending SQL rather than being silently scored as incorrect.
    """
    try:
        gold = backend.run_sql(gold_sql)
    except Exception as exc:  # noqa: BLE001 - a failing gold query means a malformed EvalCase
        raise ValueError(f"gold SQL failed to execute: {gold_sql!r}: {exc}") from exc
    try:
        cand = backend.run_sql(candidate_sql)
    except Exception:  # noqa: BLE001 - an unexecutable candidate is simply invalid
        return (False, False)
    return (True, _rows_match(cand.rows, gold.rows))


def evaluate(
    system_name: str,
    cases: list[EvalCase],
    run_one: Callable[[EvalCase, StorageBackend], tuple[str, int]],
) -> SystemReport:
    """Run `run_one(case, backend) -> (candidate_sql, attempts)` over cases; fresh DuckDB each.

    Scoring uses a SEPARATE freshly-seeded backend per case so a system-under-test run cannot leak
    state into scoring (the engine is SELECT-oriented, but this keeps the harness robust).
    """
    n = len(cases)
    valid = correct = attempts_total = 0
    for case in cases:
        run_backend = DuckDBBackend()
        run_backend.setup(case.schema_sql)
        candidate_sql, attempts = run_one(case, run_backend)
        attempts_total += attempts
        score_backend = DuckDBBackend()
        score_backend.setup(case.schema_sql)
        is_valid, is_correct = score_case(candidate_sql, case.gold_sql, score_backend)
        valid += int(is_valid)
        correct += int(is_correct)
    return SystemReport(
        name=system_name,
        n=n,
        valid_sql_rate=valid / n if n else 0.0,
        execution_accuracy=correct / n if n else 0.0,
        avg_attempts=attempts_total / n if n else 0.0,
    )
