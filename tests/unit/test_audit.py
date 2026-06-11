import json
import logging

from engine.governance.audit import AuditLog


def test_writes_one_json_line_with_fields(caplog):
    with caplog.at_level(logging.INFO, logger="asklake.audit"):
        AuditLog().write(
            user="alice",
            role="analyst",
            path="sql",
            decision="allowed",
            row_count=3,
            ms=12.0,
            question="top films",
        )
    assert len(caplog.records) == 1
    payload = json.loads(caplog.records[0].message)
    assert payload["user"] == "alice"
    assert payload["decision"] == "allowed"
    assert payload["row_count"] == 3


def test_question_is_truncated_to_120_chars(caplog):
    with caplog.at_level(logging.INFO, logger="asklake.audit"):
        AuditLog().write(
            user="u",
            role="public",
            path="sql",
            decision="blocked",
            reason="no LIMIT",
            question="x" * 500,
        )
    payload = json.loads(caplog.records[0].message)
    assert len(payload["question"]) == 120
