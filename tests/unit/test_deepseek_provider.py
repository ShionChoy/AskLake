from engine.llm.deepseek_provider import DeepSeekProvider
from engine.ports.llm import LLMProvider


class _FakeResponse:
    def __init__(self, content: str):
        self._content = content
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"role": "assistant", "content": self._content}}]}


class _FakeClient:
    """Captures the POST args and returns a canned chat-completion response."""

    def __init__(self, content: str):
        self._content = content
        self.calls: list[dict] = []

    def post(self, url, headers=None, json=None):
        self.calls.append({"url": url, "headers": headers or {}, "json": json or {}})
        return _FakeResponse(self._content)


def test_is_llm_provider():
    assert isinstance(DeepSeekProvider(api_key="x", client=_FakeClient("SELECT 1")), LLMProvider)


def test_complete_builds_messages_and_parses_content():
    client = _FakeClient("SELECT title FROM movies")
    provider = DeepSeekProvider(model="deepseek-v4-flash", api_key="secret", client=client)
    out = provider.complete("write sql", system="you are a SQL writer")
    assert out == "SELECT title FROM movies"
    call = client.calls[0]
    assert call["url"].endswith("/chat/completions")
    assert call["headers"]["Authorization"] == "Bearer secret"
    body = call["json"]
    assert body["model"] == "deepseek-v4-flash"
    assert body["messages"][0] == {"role": "system", "content": "you are a SQL writer"}
    assert body["messages"][-1] == {"role": "user", "content": "write sql"}
    assert body.get("temperature") == 0


def test_complete_without_system_has_only_user_message():
    client = _FakeClient("ok")
    DeepSeekProvider(api_key="x", client=client).complete("hello")
    assert client.calls[0]["json"]["messages"] == [{"role": "user", "content": "hello"}]
