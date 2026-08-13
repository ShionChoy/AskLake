from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SchemaProvider(Protocol):
    """Produce raw or semantic grounding context for the LLM."""

    def schema_context(self, question: str) -> str: ...
