import pytest

from engine.governance.policy import GovernanceError, Policy, PolicyGovernance, RolePolicy
from engine.governance.sql import SqlGuardrails, SqlPolicyError, analyze_read_query
from engine.governance.views import build_role_views
from engine.lakehouse.duckdb_backend import DuckDBBackend


@pytest.mark.parametrize(
    "sql,code",
    [
        ("SELECT 1; SELECT 2", "multiple_statements"),
        ("DROP TABLE movies", "statement_not_read_only"),
        ("PRAGMA version", "statement_not_read_only"),
        ("EXPLAIN SELECT * FROM movies", "statement_not_read_only"),
        ("SELECT * FROM main.movies", "schema_access_denied"),
        ("SELECT * FROM information_schema.tables", "schema_access_denied"),
        ("SELECT * FROM read_parquet('/tmp/private.parquet')", "external_access_denied"),
        ("SELECT getenv('HOME')", "external_access_denied"),
        ("SELECT file_size('/etc/passwd')", "external_access_denied"),
        (
            "WITH RECURSIVE forever(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM forever) "
            "SELECT * FROM forever",
            "query_too_complex",
        ),
        ("SELECT * FROM movies CROSS JOIN people", "cross_join_denied"),
        ("SELECT * FROM movies, people", "cross_join_denied"),
    ],
)
def test_ast_guard_rejects_boundary_bypasses(sql, code):
    with pytest.raises(SqlPolicyError) as raised:
        analyze_read_query(sql, SqlGuardrails())
    assert raised.value.code == code


def test_ast_guard_handles_ctes_without_treating_them_as_tables():
    analysis = analyze_read_query(
        "WITH selected AS (SELECT id FROM movies) SELECT * FROM selected", SqlGuardrails()
    )
    assert analysis.tables == ("movies",)


def test_nested_cte_cannot_shadow_an_outer_physical_table():
    analysis = analyze_read_query(
        "SELECT * FROM movies WHERE EXISTS (WITH movies AS (SELECT 1) SELECT * FROM movies)",
        SqlGuardrails(),
    )
    assert analysis.tables == ("movies",)


def test_role_scoping_blocks_alias_based_mask_bypass_and_caps_results():
    backend = DuckDBBackend()
    backend.setup(
        "CREATE TABLE people AS SELECT * FROM (VALUES "
        "('Alice', 1970), ('Bob', 1980)) v(primaryName, birthYear)"
    )
    policy = Policy(
        version=2,
        roles=("public",),
        role_rules={
            "public": RolePolicy(
                actions=frozenset({"ask", "raw_sql"}),
                tables=("people",),
                columns={"people.birthYear": "mask"},
                max_rows=1,
            )
        },
    )
    build_role_views(backend, policy)
    scoped = PolicyGovernance(policy).scoped_backend(backend, "public")
    result = scoped.run_sql(
        "SELECT primaryName AS person, birthYear AS definitely_not_pii FROM people ORDER BY 1"
    )
    assert result.columns == ["person", "definitely_not_pii"]
    assert result.rows == [("Alice", None)]


def test_unknown_role_and_denied_action_fail_closed():
    policy = Policy(
        version=2,
        roles=("public",),
        role_rules={"public": RolePolicy(actions=frozenset({"ask"}))},
    )
    governance = PolicyGovernance(policy, action="raw_sql")
    with pytest.raises(GovernanceError) as denied:
        governance.before_query("SELECT 1", role="public")
    assert denied.value.code == "action_denied"
    with pytest.raises(GovernanceError) as unknown:
        governance.before_query("SELECT 1", role="administrator")
    assert unknown.value.code == "unknown_role"


def test_version_two_role_defaults_grant_nothing():
    policy = Policy(
        version=2,
        roles=("accidentally_empty",),
        role_rules={"accidentally_empty": RolePolicy()},
    )
    assert not policy.allows_action("accidentally_empty", "ask")
    assert policy.tables_for("accidentally_empty", {"movies"}) == frozenset()
