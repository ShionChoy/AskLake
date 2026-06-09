from __future__ import annotations

import re
from dataclasses import dataclass

from engine.ports.storage import QueryResult

_TOPN = re.compile(r"\b(top|highest|lowest|most|least|best|worst)\b")


@dataclass(frozen=True)
class Critique:
    ok: bool
    reasons: tuple[str, ...]


def is_ranking_question(question: str) -> bool:
    """True when the question asks for a ranking/superlative (top-N) answer."""
    return bool(_TOPN.search(question.lower()))


def assess(question: str, sql: str, result: QueryResult | None, *, difficulty: str) -> Critique:
    """Correctness-level checks (deterministic, dataset-agnostic). Returns reasons to correct."""
    if result is None:
        return Critique(False, ("query did not execute",))
    reasons: list[str] = []
    if not result.rows:
        reasons.append("query returned 0 rows")
    if is_ranking_question(question):
        su = sql.upper()
        if "ORDER BY" not in su:
            reasons.append("top-N/superlative question but SQL has no ORDER BY")
        if "LIMIT" not in su:
            reasons.append("top-N/superlative question but SQL has no LIMIT")
    return Critique(not reasons, tuple(reasons))


def _signature(result: QueryResult) -> tuple:
    # Order-insensitive multiset signature (matches the eval's scoring semantics).
    return tuple(sorted(repr(r) for r in result.rows))


def select_consistent(
    executed: list[tuple[str, QueryResult]],
) -> tuple[str, QueryResult]:
    """Self-consistency: group executed candidates by result-set signature, return the first
    member of the largest group (earliest-seen group wins ties). `executed` must be non-empty."""
    groups: dict[tuple, list[tuple[str, QueryResult]]] = {}
    order: list[tuple] = []
    for sql, res in executed:
        sig = _signature(res)
        if sig not in groups:
            groups[sig] = []
            order.append(sig)
        groups[sig].append((sql, res))
    best = max(order, key=lambda s: (len(groups[s]), -order.index(s)))
    return groups[best][0]
