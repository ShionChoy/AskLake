from engine.llm.anthropic_provider import AnthropicProvider
from engine.ports.llm import LLMProvider


class _Block:
    type = "text"

    def __init__(self, text):
        self.text = text


class _Msg:
    def __init__(self, blocks):
        self.content = blocks


class _Messages:
    def __init__(self, outer):
        self._outer = outer

    def create(self, **kwargs):
        self._outer.last_kwargs = kwargs
        return _Msg([_Block("SELECT 1")])


class FakeClient:
    def __init__(self):
        self.messages = _Messages(self)
        self.last_kwargs = None


def test_anthropic_provider_satisfies_protocol_and_returns_text():
    client = FakeClient()
    provider = AnthropicProvider(model="claude-sonnet-5", client=client)
    assert isinstance(provider, LLMProvider)
    out = provider.complete("hi", system="sys")
    assert out == "SELECT 1"
    assert client.last_kwargs["model"] == "claude-sonnet-5"
    assert client.last_kwargs["system"] == "sys"
    assert client.last_kwargs["messages"][0]["content"] == "hi"
