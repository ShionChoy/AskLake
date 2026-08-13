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


def test_explicit_deepseek_key_and_model(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("ASKLAKE_LLM_PROVIDER", raising=False)
    p = make_provider("deepseek", api_key="sk-X", model="deepseek-v4-pro")
    assert type(p).__name__ == "DeepSeekProvider"
    assert p._api_key == "sk-X"
    assert p._model == "deepseek-v4-pro"


def test_explicit_deepseek_model_defaults_when_omitted(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("ASKLAKE_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("ASKLAKE_DEEPSEEK_MODEL", raising=False)
    p = make_provider("deepseek", api_key="sk-X")
    assert p._model == "deepseek-v4-flash"


def test_explicit_anthropic_key_and_model(monkeypatch):
    pytest.importorskip("anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ASKLAKE_LLM_PROVIDER", raising=False)
    p = make_provider("anthropic", api_key="sk-Y", model="claude-opus-5")
    # AnthropicProvider passes api_key straight to the SDK client (not stored on self), so we
    # assert model+class here; the DeepSeek test above covers the api_key-forwarding line.
    assert type(p).__name__ == "AnthropicProvider"
    assert p._model == "claude-opus-5"
