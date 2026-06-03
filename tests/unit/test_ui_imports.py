import importlib


def test_ui_module_imports():
    mod = importlib.import_module("ui.app")
    assert hasattr(mod, "render")
