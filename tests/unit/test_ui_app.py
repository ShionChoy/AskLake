import json

import pytest
import requests

from ui.app import DEFAULT_MODELS, ApiError, _ask_body, _creds_payload, _response_json


def test_creds_payload_omits_empty_fields():
    state = {"provider": "deepseek", "model": "", "api_key": "sk-X"}
    assert _creds_payload(state) == {"provider": "deepseek", "api_key": "sk-X"}


def test_creds_payload_keeps_all_set_fields():
    state = {"provider": "anthropic", "model": "claude-opus-5", "api_key": "sk-Y"}
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


def test_model_picker_uses_current_provider_models_and_defaults_to_flash():
    assert DEFAULT_MODELS["deepseek"] == ["deepseek-v4-flash", "deepseek-v4-pro"]
    assert DEFAULT_MODELS["anthropic"] == [
        "claude-sonnet-5",
        "claude-opus-5",
        "claude-fable-5",
        "claude-haiku-4-5",
    ]


def test_auth_headers_builds_bearer_when_token_present():
    import ui.app as app

    assert app._auth_headers({"access_token": "tok_a"}) == {"Authorization": "Bearer tok_a"}


def test_auth_headers_empty_when_no_token():
    import ui.app as app

    assert app._auth_headers({}) == {}
    assert app._auth_headers({"access_token": ""}) == {}


def _response(status: int, body: dict) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(body).encode()
    return response


def test_response_json_preserves_governance_denial_and_request_id():
    with pytest.raises(ApiError) as raised:
        _response_json(
            _response(
                403,
                {"error": "role denied", "code": "action_denied", "request_id": "req-1"},
            )
        )
    assert raised.value.status_code == 403
    assert raised.value.code == "action_denied"
    assert "req-1" in str(raised.value)


def test_response_json_does_not_treat_401_detail_as_success():
    with pytest.raises(ApiError) as raised:
        _response_json(_response(401, {"detail": "invalid bearer credentials"}))
    assert raised.value.status_code == 401
