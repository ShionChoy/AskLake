from __future__ import annotations

from engine.ports.storage import StorageBackend


class RawSchemaProvider:
    """Bare schema context: every table + columns from backend introspection.
    Ignores the question (no retrieval). Replaced by SemanticLayerProvider in P3."""

    def __init__(self, backend: StorageBackend):
        self._backend = backend

    def schema_context(self, question: str) -> str:
        lines = []
        for t in self._backend.list_tables():
            cols = ", ".join(f"{c.name} {c.type}" for c in t.columns)
            lines.append(f"- {t.name}({cols})")
        return "Tables:\n" + "\n".join(lines)
