from engine.governance.policy import Policy
from engine.governance.views import build_role_views
from engine.lakehouse.duckdb_backend import DuckDBBackend
from engine.lakehouse.role_scoped_backend import RoleScopedBackend
from engine.ports.storage import StorageBackend


def _backend() -> DuckDBBackend:
    b = DuckDBBackend()
    b.setup(
        "CREATE TABLE title_ratings AS SELECT * FROM (VALUES "
        "('tt1', 9.0, 100000), ('tt2', 8.0, 10)) v(tconst, averageRating, numVotes);"
    )
    build_role_views(
        b,
        Policy(
            roles=("analyst", "public"),
            row_security={"public": {"title_ratings": "numVotes >= 25000"}},
        ),
    )
    return b


def test_is_storage_backend():
    assert isinstance(RoleScopedBackend(_backend(), "public"), StorageBackend)


def test_public_sees_only_public_catalog():
    rb = RoleScopedBackend(_backend(), "public")
    assert rb.run_sql("SELECT tconst FROM title_ratings").rows == [("tt1",)]


def test_analyst_sees_everything():
    rb = RoleScopedBackend(_backend(), "analyst")
    assert {r[0] for r in rb.run_sql("SELECT tconst FROM title_ratings").rows} == {"tt1", "tt2"}


def test_search_path_is_set_per_call_not_leaked_across_roles():
    base = _backend()
    pub = RoleScopedBackend(base, "public")
    ana = RoleScopedBackend(base, "analyst")
    assert pub.run_sql("SELECT count(*) FROM title_ratings").rows == [(1,)]
    assert ana.run_sql("SELECT count(*) FROM title_ratings").rows == [(2,)]
    assert pub.run_sql("SELECT count(*) FROM title_ratings").rows == [(1,)]
