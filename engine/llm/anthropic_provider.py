from __future__ import annotations


class AnthropicProvider:
    """LLMProvider backed by the Anthropic Messages API. `client` is injectable for tests."""

    def __init__(self, model: str, client=None, api_key: str | None = None, max_tokens: int = 1024):
        if client is None:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
        self._client = client
        self._model = model
        self._max_tokens = max_tokens

    def complete(self, prompt: str, system: str | None = None) -> str:
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
