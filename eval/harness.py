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


@dataclass(frozen=True)
class SystemReport:
    name: str
    n: int
    valid_sql_rate: float
    execution_accuracy: float
    avg_attempts: float


def _rows_match(a, b) -> bool:
    # Execution Accuracy compares result sets as multisets (order-insensitive).
    return Counter(map(tuple, a)) == Counter(map(tuple, b))


def score_case(candidate_sql: str, gold_sql: str, backend: StorageBackend) -> tuple[bool, bool]:
    """Return (valid, correct): valid = candidate executes; correct = result multiset == gold's."""
    gold = backend.run_sql(gold_sql)
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
    """Run `run_one(case, backend) -> (candidate_sql, attempts)` over cases; fresh DuckDB each."""
    n = len(cases)
    valid = correct = attempts_total = 0
    for case in cases:
        backend = DuckDBBackend()
        backend.setup(case.schema_sql)
        candidate_sql, attempts = run_one(case, backend)
        attempts_total += attempts
        is_valid, is_correct = score_case(candidate_sql, case.gold_sql, backend)
        valid += int(is_valid)
        correct += int(is_correct)
    return SystemReport(
        name=system_name,
        n=n,
        valid_sql_rate=valid / n if n else 0.0,
        execution_accuracy=correct / n if n else 0.0,
        avg_attempts=attempts_total / n if n else 0.0,
    )
