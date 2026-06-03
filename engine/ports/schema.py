from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SchemaProvider(Protocol):
    """Produces grounding context for the LLM. Raw (P1) -> semantic layer (P3)."""

    def schema_context(self, question: str) -> str: ...
