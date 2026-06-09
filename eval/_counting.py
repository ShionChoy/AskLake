from __future__ import annotations

from engine.ports.llm import LLMProvider


class CountingLLM:
    """Wraps an LLMProvider to count `.complete()` calls — the per-case cost signal for the
    ablation eval. Delegates verbatim; records nothing about content."""

    def __init__(self, inner: LLMProvider):
        self._inner = inner
        self.calls = 0

    def complete(self, prompt: str, system: str | None = None) -> str:
        self.calls += 1
        return self._inner.complete(prompt, system=system)
