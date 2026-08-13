import pytest

from engine.governance.policy import (
    GovernanceError,
    Policy,
    PolicyGovernance,
    RowFilter,
    load_policy,
)
from engine.ports.governance import GovernanceHook
from engine.ports.storage import QueryResult


def _gov() -> PolicyGovernance:
    return PolicyGovernance(
        Policy(
            pii_columns=("birthYear",),
            mask_roles=("public",),
            row_filters={"public": (RowFilter("titleType", ("adult",)),)},
            require_limit=True,
            forbid_writes=True,
        )
    )


def _result() -> QueryResult:
    return QueryResult(
        columns=["primaryName", "birthYear", "titleType"],
        rows=[("Nolan", 1970, "movie"), ("Hidden", 1980, "adult")],
    )


def test_is_governance_hook():
    assert isinstance(_gov(), GovernanceHook)


def test_before_query_allows_safe_select_with_limit():
    sql = "SELECT primaryName FROM people LIMIT 10"
    assert _gov().before_query(sql, role="public") == sql


def test_before_query_blocks_missing_limit():
    with pytest.raises(GovernanceError, match="LIMIT"):
        _gov().before_query("SELECT primaryName FROM people", role="public")


def test_before_query_blocks_writes_and_multistatement():
    with pytest.raises(GovernanceError):
        _gov().before_query("DROP TABLE people", role="analyst")
    with pytest.raises(GovernanceError):
        _gov().before_query("SELECT 1 LIMIT 1; SELECT 2 LIMIT 1", role="analyst")


def test_after_result_analyst_sees_everything():
    out = _gov().after_result(_result(), role="analyst")
    assert out.rows == [("Nolan", 1970, "movie"), ("Hidden", 1980, "adult")]


def test_after_result_public_is_filtered_and_masked():
    out = _gov().after_result(_result(), role="public")
    # 'adult' row filtered out; birthYear PII masked
    assert out.columns == ["primaryName", "birthYear", "titleType"]
    assert out.rows == [("Nolan", "***", "movie")]


def test_load_policy_reads_roles_and_row_security(tmp_path):
    p = tmp_path / "gov.yaml"
    p.write_text(
        "roles:\n"
        "  analyst: {pii: allow}\n"
        "  public:  {pii: mask}\n"
        "pii_columns: [birthYear]\n"
        "row_security:\n"
        "  public:\n"
        '    title_ratings: "numVotes >= 25000"\n'
        "cost_guardrail: {require_limit: true, forbid_writes: true}\n"
    )
    pol = load_policy(p)
    assert set(pol.roles) == {"analyst", "public"}
    assert pol.row_security == {"public": {"title_ratings": "numVotes >= 25000"}}
    assert pol.mask_roles == ("public",)  # unchanged behavior


def test_load_policy_defaults_when_absent(tmp_path):
    p = tmp_path / "gov.yaml"
    p.write_text("pii_columns: [birthYear]\n")
    pol = load_policy(p)
    assert pol.roles == ()
    assert pol.row_security == {}


def test_version_two_yaml_role_defaults_grant_nothing(tmp_path):
    path = tmp_path / "governance.yaml"
    path.write_text("version: 2\nanonymous_role: public\nroles:\n  public: {}\n")
    policy = load_policy(path)
    assert not policy.allows_action("public", "ask")
    assert policy.tables_for("public", {"movies"}) == frozenset()


def test_version_two_yaml_rejects_wildcard_or_unclassified_grants(tmp_path):
    path = tmp_path / "governance.yaml"
    path.write_text(
        "version: 2\nanonymous_role: public\nroles:\n  public: {actions: [ask], tables: ['*']}\n"
    )
    with pytest.raises(ValueError, match="must enumerate"):
        load_policy(path)

    path.write_text(
        "version: 2\nanonymous_role: public\nroles:\n"
        "  public: {actions: [ask], tables: [unclassified]}\n"
    )
    with pytest.raises(ValueError, match="unclassified tables"):
        load_policy(path)
