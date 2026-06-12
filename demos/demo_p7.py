"""Phase 7 demo: authenticated, role-based access control (hermetic).

Same question, two callers: an analyst token sees the full catalog with real birth years;
an anonymous caller (no token) degrades to `public` -> only the public catalog (popular
titles) with birth years masked. Plus one cost-guardrail block. No network, no API key."""

from __future__ import annotations

from engine.governance.policy import GovernanceError, Policy, PolicyGovernance
from engine.governance.views import build_role_views
from engine.lakehouse.duckdb_backend import DuckDBBackend
from engine.lakehouse.role_scoped_backend import RoleScopedBackend

_SEED = (
    "CREATE TABLE title_ratings AS SELECT * FROM (VALUES "
    "('tt1', 9.0, 100000), ('tt2', 8.5, 50000), ('tt3', 7.0, 10)) "
    "v(tconst, averageRating, numVotes);"
    "CREATE TABLE name_basics AS SELECT * FROM (VALUES "
    "('nm1','Nolan',1970)) v(nconst, primaryName, birthYear);"
)

_POLICY = Policy(
    roles=("analyst", "public"),
    pii_columns=("birthYear",),
    mask_roles=("public",),
    row_security={"public": {"title_ratings": "numVotes >= 25000"}},
    require_limit=True,
    forbid_writes=True,
)


def _count(role: str) -> int:
    b = DuckDBBackend()
    b.setup(_SEED)
    build_role_views(b, _POLICY)
    rb = RoleScopedBackend(b, role)
    return len(rb.run_sql("SELECT tconst FROM title_ratings").rows)


def _birthyear(role: str):
    b = DuckDBBackend()
    b.setup(_SEED)
    build_role_views(b, _POLICY)
    rb = RoleScopedBackend(b, role)
    return rb.run_sql("SELECT birthYear FROM name_basics").rows[0][0]


def run_demo_p7() -> dict:
    blocked = False
    try:
        PolicyGovernance(_POLICY).before_query("SELECT * FROM title_ratings", role="public")
    except GovernanceError:
        blocked = True
    return {
        "analyst_rows": _count("analyst"),
        "public_rows": _count("public"),
        "public_birthyear_masked": _birthyear("public") is None,
        "blocked_without_limit": blocked,
    }


if __name__ == "__main__":
    out = run_demo_p7()
    print("analyst sees rows:", out["analyst_rows"])
    print("public sees rows: ", out["public_rows"], "(row-level security)")
    print("public birthYear masked:", out["public_birthyear_masked"])
    print("query without LIMIT blocked:", out["blocked_without_limit"])
    print("demo-p7 OK")
