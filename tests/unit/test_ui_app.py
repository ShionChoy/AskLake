from ui.app import _ask_body, _creds_payload


def test_creds_payload_omits_empty_fields():
    state = {"provider": "deepseek", "model": "", "api_key": "sk-X"}
    assert _creds_payload(state) == {"provider": "deepseek", "api_key": "sk-X"}


def test_creds_payload_keeps_all_set_fields():
    state = {"provider": "anthropic", "model": "claude-opus-4-8", "api_key": "sk-Y"}
    assert _creds_payload(state) == state


def test_creds_payload_empty_when_unset():
    assert _creds_payload({}) == {}


def test_ask_body_includes_question_path_and_creds():
    state = {"path": "graph", "provider": "deepseek", "model": "", "api_key": "sk-X"}
    assert _ask_body("hello", state) == {
        "question": "hello",
        "path": "graph",
        "provider": "deepseek",
        "api_key": "sk-X",
    }


def test_ask_body_defaults_path_to_auto():
    assert _ask_body("hi", {})["path"] == "auto"


def test_auth_headers_builds_bearer_when_token_present():
    import ui.app as app

    assert app._auth_headers({"access_token": "tok_a"}) == {"Authorization": "Bearer tok_a"}


def test_auth_headers_empty_when_no_token():
    import ui.app as app

    assert app._auth_headers({}) == {}
    assert app._auth_headers({"access_token": ""}) == {}
