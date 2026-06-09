from __future__ import annotations

import re
from dataclasses import dataclass, field

from engine.ports.storage import StorageBackend
from engine.semantic.semantic_model import SemanticLayer

_WORD = re.compile(r"[a-z0-9]+")
_SPLIT = re.compile(r"[,;|]")
# Consecutive Capitalized words -> a candidate named entity ("Keanu Reeves").
_CAP_SPAN = re.compile(r"[A-Z][\w.&'-]*(?:\s+[A-Z][\w.&'-]*)*")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


@dataclass(frozen=True)
class ValueHint:
    column: str
    table: str
    value: str
    mode: str  # "categorical" | "entity"


@dataclass
class ValueIndex:
    """Links question entities to real stored values. Categorical values are pre-indexed;
    entity columns are probed against the backend at link time (not enumerated)."""

    # column -> (table, frozenset of atomic distinct values)
    categorical: dict[str, tuple[str, frozenset[str]]] = field(default_factory=dict)
    # column -> table (probed lazily)
    entity_columns: dict[str, str] = field(default_factory=dict)
    backend: StorageBackend | None = None
    probe_limit: int = 5

    def link(self, question: str) -> list[ValueHint]:
        hints: list[ValueHint] = []
        qtok = _tokens(question)
        for col, (table, values) in self.categorical.items():
            for v in values:
                vtok = _tokens(v)
                if len(v) >= 2 and vtok and vtok <= qtok:
                    hints.append(ValueHint(col, table, v, "categorical"))
        if self.entity_columns and self.backend is not None:
            spans = [s.strip() for s in _CAP_SPAN.findall(question) if _tokens(s)]
            for col, table in self.entity_columns.items():
                for span in spans:
                    for value in self._probe(table, col, span):
                        hints.append(ValueHint(col, table, value, "entity"))
        return hints

    def _probe(self, table: str, col: str, span: str) -> list[str]:
        sql = (
            f'SELECT DISTINCT "{col}" FROM "{table}" '
            f"WHERE \"{col}\" ILIKE '%{span.replace(chr(39), chr(39) * 2)}%' "
            f"LIMIT {int(self.probe_limit)}"
        )
        try:
            res = self.backend.run_sql(sql)  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001 - a bad probe simply yields no hint
            return []
        return [r[0] for r in res.rows if r and r[0] is not None]


def build_value_index(
    layer: SemanticLayer, backend: StorageBackend, max_distinct: int = 2000
) -> ValueIndex:
    idx = ValueIndex(backend=backend)
    for t in layer.tables:
        for c in t.columns:
            mode = c.link
            if mode == "auto":
                mode = _auto_mode(backend, t.name, c.name)
            if mode == "categorical":
                idx.categorical[c.name] = (
                    t.name,
                    _distinct_atoms(backend, t.name, c.name, max_distinct),
                )
            elif mode == "entity":
                idx.entity_columns[c.name] = t.name
    return idx


def _distinct_atoms(backend, table, col, cap):
    try:
        res = backend.run_sql(f'SELECT DISTINCT "{col}" FROM "{table}" LIMIT {int(cap)}')
    except Exception:  # noqa: BLE001
        return frozenset()
    atoms: set[str] = set()
    for row in res.rows:
        if not row or row[0] is None:
            continue
        for part in _SPLIT.split(str(row[0])):
            part = part.strip()
            if part:
                atoms.add(part)
    return frozenset(atoms)


def _auto_mode(backend, table, col) -> str:
    try:
        res = backend.run_sql(f'SELECT COUNT(DISTINCT "{col}") FROM "{table}"')
        n = res.rows[0][0] if res.rows else 0
    except Exception:  # noqa: BLE001
        return ""
    return "categorical" if n is not None and n <= 50 else "entity"


def format_hints(hints: list[ValueHint]) -> str:
    """Render hints as a context block of concrete predicates for the SQL writer."""
    lines: list[str] = []
    for h in hints:
        safe = h.value.replace("'", "''")
        if h.mode == "categorical":
            lines.append(f"- {h.column} LIKE '%{safe}%'")
        else:
            lines.append(f"- {h.column} = '{safe}'")
    return "\n".join(lines)
