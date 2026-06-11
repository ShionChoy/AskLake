from __future__ import annotations

import json
import logging

_LOGGER = logging.getLogger("asklake.audit")


class AuditLog:
    """Lightweight, non-persistent audit sink: one structured JSON line per query.

    Not a port; an additive seam. Never logs secrets (callers must pre-redact)."""

    def __init__(self, logger: logging.Logger | None = None):
        self._log = logger or _LOGGER

    def write(self, *, question: str = "", **fields) -> None:
        record = {"question": question[:120], **fields}
        self._log.info(json.dumps(record, default=str))
