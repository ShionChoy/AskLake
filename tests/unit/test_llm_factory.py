import pytest

from engine.llm.factory import make_provider


def test_auto_selects_deepseek_when_key_present(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    monkeypatch.delenv("ASKLAKE_LLM_PROVIDER", raising=False)
    assert type(make_provider()).__name__ == "DeepSeekProvider"


def test_raises_when_nothing_configured(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ASKLAKE_LLM_PROVIDER", raising=False)
    with pytest.raises(RuntimeError):
        make_provider()
