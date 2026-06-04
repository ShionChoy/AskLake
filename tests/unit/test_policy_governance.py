import pytest

from engine.governance.policy import GovernanceError, Policy, PolicyGovernance, RowFilter
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
    with pytest.raises(GovernanceError, match="guardrail"):
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
