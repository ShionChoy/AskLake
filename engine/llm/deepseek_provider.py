from __future__ import annotations

import os
from typing import Any

import httpx

_DEFAULT_BASE_URL = "https://api.deepseek.com"


class DeepSeekProvider:
    """LLMProvider backed by DeepSeek's OpenAI-compatible chat-completions API (via httpx).

    The LLM is a swappable component behind the LLMProvider port; this is a sibling of
    AnthropicProvider for the headline eval run. The HTTP client is injectable so tests run
    without network or a key. Set DEEPSEEK_API_KEY in the environment for real use."""

    def __init__(
        self,
        model: str = "deepseek-v4-flash",
        api_key: str | None = None,
        base_url: str = _DEFAULT_BASE_URL,
        client: Any = None,
        timeout: float = 60.0,
    ):
        self._model = model
        self._api_key = api_key if api_key is not None else os.environ.get("DEEPSEEK_API_KEY", "")
        self._base_url = base_url.rstrip("/")
        self._client = client
        self._timeout = timeout

    def complete(self, prompt: str, system: str | None = None) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": 0,
            "stream": False,
        }
        client = self._client or httpx.Client(timeout=self._timeout)
        resp = client.post(
            f"{self._base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"] or ""
