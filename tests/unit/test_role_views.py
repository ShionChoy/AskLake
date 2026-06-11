from engine.governance.policy import Policy
from engine.governance.views import build_role_views
from engine.lakehouse.duckdb_backend import DuckDBBackend


def _backend() -> DuckDBBackend:
    b = DuckDBBackend()
    b.setup(
        "CREATE TABLE title_ratings AS SELECT * FROM (VALUES "
        "('tt1', 9.0, 100000), ('tt2', 8.0, 10)) v(tconst, averageRating, numVotes);"
        "CREATE TABLE name_basics AS SELECT * FROM (VALUES "
        "('nm1','Nolan',1970), ('nm2','Hidden',1980)) v(nconst, primaryName, birthYear);"
    )
    return b


def _policy() -> Policy:
    return Policy(
        roles=("analyst", "public"),
        pii_columns=("birthYear",),
        mask_roles=("public",),
        row_security={"public": {"title_ratings": "numVotes >= 25000"}},
    )


def test_public_view_filters_rows_and_masks_pii():
    b = _backend()
    build_role_views(b, _policy())
    b.run_sql("SET search_path='rls_public'")
    ratings = b.run_sql("SELECT tconst FROM title_ratings").rows
    assert ratings == [("tt1",)]  # tt2 (10 votes) filtered out
    names = b.run_sql("SELECT primaryName, birthYear FROM name_basics").rows
    assert names == [("Nolan", None), ("Hidden", None)]  # birthYear redacted to NULL


def test_analyst_view_is_passthrough():
    b = _backend()
    build_role_views(b, _policy())
    b.run_sql("SET search_path='rls_analyst'")
    assert {r[0] for r in b.run_sql("SELECT tconst FROM title_ratings").rows} == {"tt1", "tt2"}
    assert b.run_sql("SELECT birthYear FROM name_basics WHERE nconst='nm1'").rows == [(1970,)]


def test_build_is_idempotent():
    b = _backend()
    build_role_views(b, _policy())
    build_role_views(b, _policy())  # CREATE OR REPLACE -> no error
    b.run_sql("SET search_path='rls_public'")
    assert b.run_sql("SELECT count(*) FROM title_ratings").rows == [(1,)]


def test_predicate_referencing_missing_table_falls_back_to_passthrough():
    # public predicate on title_basics references main.title_ratings, which is ABSENT here.
    # build_role_views must not raise at boot: it falls back to a pass-through view.
    b = DuckDBBackend()
    b.setup("CREATE TABLE title_basics AS SELECT * FROM (VALUES ('tt1'),('tt2')) v(tconst);")
    build_role_views(b, Policy(
        roles=("public",),
        row_security={"public": {"title_basics":
            "tconst IN (SELECT tconst FROM main.title_ratings WHERE numVotes >= 25000)"}},
    ))
    b.run_sql("SET search_path='rls_public'")
    assert {r[0] for r in b.run_sql("SELECT tconst FROM title_basics").rows} == {"tt1", "tt2"}
