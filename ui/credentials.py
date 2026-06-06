"""Local, UI-side store for the user's API credentials.

A single JSON file at ~/.config/asklake/credentials.json (dir 0700, file 0600), on the user's
own machine. The API server never reads or writes this — the UI loads it to pre-fill the sidebar
and sends the key per request. Stdlib only; no Streamlit import (keeps it unit-testable)."""

from __future__ import annotations

import json
import os
from pathlib import Path

_FILENAME = "credentials.json"


def _dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "asklake"


def path() -> Path:
    """Absolute path to the credentials file (honors $XDG_CONFIG_HOME)."""
    return _dir() / _FILENAME


def load() -> dict[str, str]:
    """Return {"provider","model","api_key"} or {} if the file is absent/unreadable."""
    try:
        data = json.loads(path().read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save(provider: str, model: str, api_key: str) -> None:
    """Write the credentials to a 0600 file under a 0700 dir on this machine."""
    d = _dir()
    d.mkdir(parents=True, exist_ok=True)
    d.chmod(0o700)
    p = path()
    payload = json.dumps({"provider": provider, "model": model, "api_key": api_key})
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, payload.encode())
    finally:
        os.close(fd)
    p.chmod(0o600)


def delete() -> None:
    """Remove the credentials file if present; a no-op when already absent."""
    path().unlink(missing_ok=True)
