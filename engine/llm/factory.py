from __future__ import annotations

import os

from engine.ports.llm import LLMProvider


def make_provider(provider: str | None = None) -> LLMProvider:
    """Construct an LLMProvider from the environment (the LLM is swappable behind the port).

    Resolution: explicit `provider` arg, else ASKLAKE_LLM_PROVIDER, else auto — DeepSeek when
    DEEPSEEK_API_KEY is set, else Anthropic when ANTHROPIC_API_KEY is set. Raises RuntimeError
    when nothing is configured."""
    choice = (provider or os.environ.get("ASKLAKE_LLM_PROVIDER") or "").lower()

    if choice == "deepseek" or (not choice and os.environ.get("DEEPSEEK_API_KEY")):
        from engine.llm.deepseek_provider import DeepSeekProvider

        return DeepSeekProvider(
            model=os.environ.get("ASKLAKE_DEEPSEEK_MODEL", "deepseek-chat"), timeout=120.0
        )
    if choice == "anthropic" or (not choice and os.environ.get("ANTHROPIC_API_KEY")):
        from engine.llm.anthropic_provider import AnthropicProvider
        from engine.settings import get_settings

        return AnthropicProvider(model=get_settings().llm_model)
    raise RuntimeError(
        "No LLM provider configured. Set DEEPSEEK_API_KEY (or ANTHROPIC_API_KEY); "
        "optionally set ASKLAKE_LLM_PROVIDER=deepseek|anthropic."
    )
