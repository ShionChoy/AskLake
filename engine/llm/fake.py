# engine/llm/fake.py
from __future__ import annotations


class FakeLLMProvider:
    """Deterministic LLM for tests. Cycles through `responses` and records prompts."""

    def __init__(self, responses: list[str]):
        if not responses:
            raise ValueError("FakeLLMProvider needs at least one response")
        self._responses = responses
        self._i = 0
        self.prompts: list[str] = []

    def complete(self, prompt: str, system: str | None = None) -> str:
        self.prompts.append(prompt)
        resp = self._responses[self._i % len(self._responses)]
        self._i += 1
        return resp
