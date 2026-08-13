from __future__ import annotations

import re

from engine.governance.policy import Policy
from engine.semantic.semantic_model import SemanticLayer, TableDef


def _mentions(text: str, names: set[str]) -> bool:
    folded = text.casefold()
    return any(re.search(rf"\b{re.escape(name.casefold())}\b", folded) for name in names)


def governed_semantic_layer(
    layer: SemanticLayer,
    policy: Policy,
    role: str,
    *,
    available_tables: set[str] | frozenset[str],
) -> SemanticLayer:
    """Remove inaccessible resources from the context sent to an external LLM."""

    allowed = policy.tables_for(role, available_tables)
    hidden_columns: set[str] = set()
    tables: list[TableDef] = []
    for table in layer.tables:
        if table.name not in allowed:
            continue
        visible = tuple(
            column
            for column in table.columns
            if policy.column_handling(role, table.name, column.name) == "allow"
        )
        hidden_columns.update(column.name for column in table.columns if column not in visible)
        tables.append(TableDef(table.name, table.description, visible))

    disallowed_tables = {table.name for table in layer.tables} - allowed
    blocked_names = hidden_columns | disallowed_tables
    metrics = tuple(
        metric for metric in layer.metrics if not _mentions(metric.expression, blocked_names)
    )
    synonyms = {
        term: target
        for term, target in layer.synonyms.items()
        if not _mentions(target, blocked_names)
    }
    few_shots = tuple(
        example for example in layer.few_shots if not _mentions(example.sql, blocked_names)
    )
    return SemanticLayer(
        tables=tuple(tables),
        metrics=metrics,
        synonyms=synonyms,
        few_shots=few_shots,
    )
