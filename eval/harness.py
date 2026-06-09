from __future__ import annotations

import math
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
    avg_llm_calls: float = 0.0  # mean LLM .complete() calls per case (cost signal)
    avg_wall_ms: float = 0.0  # mean wall-clock ms per case (latency signal)


_REL_TOL = 1e-3
_ABS_TOL = 1e-9


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _whole(v) -> bool:
    return _is_number(v) and float(v).is_integer()


def _cell_eq(a, b) -> bool:
    if _is_number(a) and _is_number(b):
        if _whole(a) and _whole(b):
            return float(a) == float(b)  # counts / years: exact, never masked
        return math.isclose(float(a), float(b), rel_tol=_REL_TOL, abs_tol=_ABS_TOL)
    return a == b


def _row_eq(r1: tuple, r2: tuple) -> bool:
    return len(r1) == len(r2) and all(_cell_eq(x, y) for x, y in zip(r1, r2, strict=False))


def _rows_match(a, b) -> bool:
    """Execution Accuracy: order-insensitive multiset match. Numeric cells compare with a
    relative tolerance (rel_tol 1e-3) so a correct aggregate scored at a different precision
    still matches; whole numbers (counts/years) and non-numeric cells compare exactly, so real
    errors (off-by-one counts, wrong labels, wrong cardinality) are never masked."""
    rows_a = [tuple(r) for r in a]
    rows_b = [tuple(r) for r in b]
    if len(rows_a) != len(rows_b):
        return False
    used = [False] * len(rows_b)
    for r1 in rows_a:
        for j, r2 in enumerate(rows_b):
            if not used[j] and _row_eq(r1, r2):
                used[j] = True
                break
        else:
            return False
    return True


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
