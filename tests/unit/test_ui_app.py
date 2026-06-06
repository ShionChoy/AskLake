from ui.app import _creds_payload


def test_creds_payload_omits_empty_fields():
    state = {"provider": "deepseek", "model": "", "api_key": "sk-X"}
    assert _creds_payload(state) == {"provider": "deepseek", "api_key": "sk-X"}


def test_creds_payload_keeps_all_set_fields():
    state = {"provider": "anthropic", "model": "claude-opus-4-8", "api_key": "sk-Y"}
    assert _creds_payload(state) == state


def test_creds_payload_empty_when_unset():
    assert _creds_payload({}) == {}
