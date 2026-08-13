from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Swappable LLM interface implemented by cloud providers and the test fake."""

    def complete(self, prompt: str, system: str | None = None) -> str: ...
