from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Swappable LLM. Implementations: Anthropic (P1), Ollama (optional)."""

    def complete(self, prompt: str, system: str | None = None) -> str: ...
