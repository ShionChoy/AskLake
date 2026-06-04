from __future__ import annotations

from pathlib import Path

from engine.semantic.retriever import LexicalSchemaRetriever, SchemaRetriever
from engine.semantic.semantic_model import SemanticLayer, load_semantic_layer


class SemanticLayerProvider:
    """SchemaProvider (P3): grounds the LLM with a curated semantic layer (table/column
    descriptions, metrics, synonyms, few-shot examples), pruned to the question by a pluggable
    SchemaRetriever. Additive sibling of RawSchemaProvider (P1, kept as the eval baseline)."""

    def __init__(self, layer: SemanticLayer, retriever: SchemaRetriever | None = None):
        self._layer = layer
        self._retriever = retriever or LexicalSchemaRetriever()

    @classmethod
    def from_yaml(
        cls, path: str | Path, retriever: SchemaRetriever | None = None
    ) -> SemanticLayerProvider:
        return cls(load_semantic_layer(path), retriever)

    def schema_context(self, question: str) -> str:
        ctx = self._retriever.select(question, self._layer)
        lines: list[str] = ["Tables:"]
        for t in ctx.tables:
            cols = ", ".join(
                c.name
                + (f" {c.type}" if c.type else "")
                + (f" -- {c.description}" if c.description else "")
                for c in t.columns
            )
            suffix = f"  # {t.description}" if t.description else ""
            lines.append(f"- {t.name}({cols}){suffix}")
        if self._layer.metrics:
            lines.append("\nMetrics:")
            for m in self._layer.metrics:
                lines.append(
                    f"- {m.name}: {m.expression}"
                    + (f"  # {m.description}" if m.description else "")
                )
        if self._layer.synonyms:
            syn = ", ".join(f"{k} -> {v}" for k, v in self._layer.synonyms.items())
            lines.append(f"\nSynonyms: {syn}")
        if ctx.few_shots:
            lines.append("\nExamples:")
            for f in ctx.few_shots:
                lines.append(f"Q: {f.question}\nSQL: {f.sql}")
        return "\n".join(lines)
