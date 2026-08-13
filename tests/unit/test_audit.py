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


def test_question_is_hashed_and_not_logged_by_default(caplog):
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
    assert payload["query_length"] == 500
    assert len(payload["query_sha256"]) == 64
    assert "question" not in payload and "query_preview" not in payload


def test_persistent_audit_file_is_owner_only_and_jsonl(tmp_path):
    path = tmp_path / "audit" / "events.jsonl"
    AuditLog(path=path).write(event="query", decision="allowed", question="private query")
    assert path.stat().st_mode & 0o777 == 0o600
    payload = json.loads(path.read_text())
    assert payload["decision"] == "allowed"
    assert "private query" not in path.read_text()
