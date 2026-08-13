from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger("asklake.audit")


class AuditLog:
    """Structured audit events with optional persistent JSONL output.

    Query/question content is hashed by default so logs remain useful for correlation without
    becoming a second sensitive dataset. The file sink is append-only between bounded rotations
    and is created with owner-only permissions. Production operators can ship the JSONL stream to
    their immutable audit platform without changing application code.
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        *,
        path: str | Path | None = None,
        include_query_text: bool = False,
        max_bytes: int = 20 * 1024 * 1024,
        backups: int = 10,
    ) -> None:
        self._log = logger or _LOGGER
        self._path = Path(path) if path else None
        self._include_query_text = include_query_text
        self._max_bytes = max_bytes
        self._backups = backups
        self._lock = threading.Lock()
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(
                self._path,
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                0o600,
            )
            os.close(descriptor)
            os.chmod(self._path, 0o600)

    def _rotate(self, incoming: int) -> None:
        if self._path is None or not self._path.exists():
            return
        if self._path.stat().st_size + incoming <= self._max_bytes:
            return
        if self._backups <= 0:
            self._path.unlink(missing_ok=True)
            return
        oldest = self._path.with_name(f"{self._path.name}.{self._backups}")
        oldest.unlink(missing_ok=True)
        for index in range(self._backups - 1, 0, -1):
            source = self._path.with_name(f"{self._path.name}.{index}")
            if source.exists():
                source.replace(self._path.with_name(f"{self._path.name}.{index + 1}"))
        self._path.replace(self._path.with_name(f"{self._path.name}.1"))

    def _append(self, line: str) -> None:
        if self._path is None:
            return
        payload = (line + "\n").encode("utf-8")
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._rotate(len(payload))
            descriptor = os.open(
                self._path,
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                0o600,
            )
            try:
                os.write(descriptor, payload)
            finally:
                os.close(descriptor)

    def write(
        self,
        *,
        question: str = "",
        query_text: str | None = None,
        **fields: Any,
    ) -> None:
        content = query_text if query_text is not None else question
        record: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "event_id": str(uuid.uuid4()),
            **fields,
        }
        if content:
            record["query_sha256"] = hashlib.sha256(content.encode("utf-8")).hexdigest()
            record["query_length"] = len(content)
            if self._include_query_text:
                record["query_preview"] = content[:120]
        line = json.dumps(record, default=str, ensure_ascii=False, separators=(",", ":"))
        self._log.info(line)
        self._append(line)
