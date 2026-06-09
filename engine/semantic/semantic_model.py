from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ColumnDef:
    name: str
    description: str = ""
    type: str = ""
    link: str = ""  # "" | "categorical" | "entity" | "auto" — drives value-linking


@dataclass(frozen=True)
class TableDef:
    name: str
    description: str = ""
    columns: tuple[ColumnDef, ...] = ()


@dataclass(frozen=True)
class MetricDef:
    name: str
    expression: str
    description: str = ""


@dataclass(frozen=True)
class FewShot:
    question: str
    sql: str


@dataclass(frozen=True)
class SemanticLayer:
    """Curated business semantics for one dataset (loaded from datasets/<name>/semantic.yaml)."""

    tables: tuple[TableDef, ...] = ()
    metrics: tuple[MetricDef, ...] = ()
    synonyms: dict[str, str] = field(default_factory=dict)  # term -> canonical column/table
    few_shots: tuple[FewShot, ...] = ()


def load_semantic_layer(path: str | Path) -> SemanticLayer:
    data = yaml.safe_load(Path(path).read_text()) or {}
    tables = tuple(
        TableDef(
            name=t["name"],
            description=t.get("description", ""),
            columns=tuple(
                ColumnDef(
                    c["name"],
                    c.get("description", ""),
                    c.get("type", ""),
                    c.get("link", ""),
                )
                for c in t.get("columns", [])
            ),
        )
        for t in data.get("tables", [])
    )
    metrics = tuple(
        MetricDef(m["name"], m["expression"], m.get("description", ""))
        for m in data.get("metrics", [])
    )
    few_shots = tuple(FewShot(f["question"], f["sql"]) for f in data.get("few_shots", []))
    return SemanticLayer(
        tables=tables,
        metrics=metrics,
        synonyms=dict(data.get("synonyms", {}) or {}),
        few_shots=few_shots,
    )
