from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from engine.semantic.semantic_model import FewShot, SemanticLayer, TableDef

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


@dataclass(frozen=True)
class RetrievedContext:
    tables: tuple[TableDef, ...]
    few_shots: tuple[FewShot, ...]


@runtime_checkable
class SchemaRetriever(Protocol):
    """Selects the question-relevant slice of a SemanticLayer. Lexical now; Qdrant-backed later."""

    def select(self, question: str, layer: SemanticLayer) -> RetrievedContext: ...


class LexicalSchemaRetriever:
    """In-process, dependency-free retrieval: rank tables/few-shots by token overlap with the
    question (expanded via the layer's synonyms). Hermetic stand-in for Qdrant vector retrieval,
    which can be added later as another SchemaRetriever without touching callers."""

    def __init__(self, max_tables: int = 0, max_few_shots: int = 3):
        self._max_tables = max_tables  # 0 = no cap
        self._max_few_shots = max_few_shots

    def _expand(self, question: str, layer: SemanticLayer) -> set[str]:
        toks = _tokens(question)
        extra: set[str] = set()
        for term, canonical in layer.synonyms.items():
            if term.lower() in toks:
                extra |= _tokens(canonical)
        return toks | extra

    def select(self, question: str, layer: SemanticLayer) -> RetrievedContext:
        qtok = self._expand(question, layer)

        def table_score(t: TableDef) -> int:
            text = " ".join(
                [t.name, t.description, *(f"{c.name} {c.description}" for c in t.columns)]
            )
            return len(qtok & _tokens(text))

        ranked = sorted(layer.tables, key=table_score, reverse=True)
        matched = [t for t in ranked if table_score(t) > 0]
        tables = tuple(matched or layer.tables)  # fall back to all tables if nothing overlaps
        if self._max_tables:
            tables = tables[: self._max_tables]

        def fs_score(f: FewShot) -> int:
            return len(qtok & _tokens(f.question))

        fs_ranked = sorted(layer.few_shots, key=fs_score, reverse=True)
        few = tuple(f for f in fs_ranked if fs_score(f) > 0)[: self._max_few_shots]
        if not few:
            few = tuple(layer.few_shots[: self._max_few_shots])
        return RetrievedContext(tables=tables, few_shots=few)
