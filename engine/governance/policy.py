from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from engine.ports.storage import QueryResult

MASK = "***"

_LIMIT = re.compile(r"\blimit\b", re.IGNORECASE)
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|copy|truncate|replace|pragma)\b",
    re.IGNORECASE,
)


class GovernanceError(Exception):
    """Raised when a query violates governance policy (write or cost guardrail)."""


@dataclass(frozen=True)
class RowFilter:
    column: str
    deny_values: tuple[str, ...]


@dataclass(frozen=True)
class Policy:
    pii_columns: tuple[str, ...] = ()
    mask_roles: tuple[str, ...] = ()  # roles whose PII columns are masked
    row_filters: dict[str, tuple[RowFilter, ...]] = field(default_factory=dict)  # role -> filters
    roles: tuple[str, ...] = ()
    row_security: dict[str, dict[str, str]] = field(
        default_factory=dict
    )  # role -> {table: predicate}
    require_limit: bool = False
    forbid_writes: bool = True


def load_policy(path: str | Path) -> Policy:
    data = yaml.safe_load(Path(path).read_text()) or {}
    roles = data.get("roles", {}) or {}
    mask_roles = tuple(r for r, cfg in roles.items() if (cfg or {}).get("pii") == "mask")
    row_filters = {
        role: tuple(
            RowFilter(f["column"], tuple(str(v) for v in f.get("deny_values", [])))
            for f in (filters or [])
        )
        for role, filters in (data.get("row_filters", {}) or {}).items()
    }
    row_security = {
        role: {str(tbl): str(pred) for tbl, pred in (tables or {}).items()}
        for role, tables in (data.get("row_security", {}) or {}).items()
    }
    guard = data.get("cost_guardrail", {}) or {}
    return Policy(
        pii_columns=tuple(data.get("pii_columns", []) or []),
        mask_roles=mask_roles,
        row_filters=row_filters,
        roles=tuple(roles.keys()),
        row_security=row_security,
        require_limit=bool(guard.get("require_limit", False)),
        forbid_writes=bool(guard.get("forbid_writes", True)),
    )


class PolicyGovernance:
    """GovernanceHook (P3): config-driven RBAC / PII / cost guardrails. Additive sibling of
    PassthroughGovernance (P0).

    before_query -> safety + cost guardrail (raises GovernanceError to block).
    after_result -> row-level filtering + column-level PII masking, by role.
    The Policy comes from datasets/<name>/governance.yaml; the engine stays dataset-agnostic."""

    def __init__(self, policy: Policy):
        self._p = policy

    @classmethod
    def from_yaml(cls, path: str | Path) -> PolicyGovernance:
        return cls(load_policy(path))

    def before_query(self, sql: str, role: str) -> str:
        body = sql.strip().rstrip(";")
        if self._p.forbid_writes and (_FORBIDDEN.search(body) or ";" in body):
            raise GovernanceError("query blocked: only a single read-only SELECT is permitted")
        if self._p.require_limit and not _LIMIT.search(body):
            raise GovernanceError("query blocked by cost guardrail: a LIMIT clause is required")
        return sql

    def after_result(self, result: QueryResult, role: str) -> QueryResult:
        rows = [list(r) for r in result.rows]
        for rf in self._p.row_filters.get(role, ()):
            if rf.column in result.columns:
                idx = result.columns.index(rf.column)
                rows = [r for r in rows if str(r[idx]) not in rf.deny_values]
        if role in self._p.mask_roles:
            for col in self._p.pii_columns:
                if col in result.columns:
                    idx = result.columns.index(col)
                    for r in rows:
                        r[idx] = MASK
        return QueryResult(columns=list(result.columns), rows=[tuple(r) for r in rows])
