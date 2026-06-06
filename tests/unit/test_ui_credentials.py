import stat

from ui import credentials


def _isolate(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))


def test_save_then_load_roundtrips(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    credentials.save("deepseek", "deepseek-chat", "sk-X")
    assert credentials.load() == {
        "provider": "deepseek",
        "model": "deepseek-chat",
        "api_key": "sk-X",
    }


def test_saved_file_is_0600(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    credentials.save("deepseek", "deepseek-chat", "sk-X")
    mode = stat.S_IMODE(credentials.path().stat().st_mode)
    assert mode == 0o600
    dir_mode = stat.S_IMODE(credentials.path().parent.stat().st_mode)
    assert dir_mode == 0o700


def test_load_missing_returns_empty(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    assert credentials.load() == {}


def test_load_corrupt_json_returns_empty(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    credentials.path().parent.mkdir(parents=True, exist_ok=True)
    credentials.path().write_text("{not valid json")
    assert credentials.load() == {}


def test_load_non_dict_json_returns_empty(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    credentials.path().parent.mkdir(parents=True, exist_ok=True)
    credentials.path().write_text("[1, 2, 3]")
    assert credentials.load() == {}


def test_delete_removes_and_is_idempotent(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    credentials.save("deepseek", "deepseek-chat", "sk-X")
    assert credentials.path().exists()
    credentials.delete()
    assert not credentials.path().exists()
    credentials.delete()  # no error when already gone
