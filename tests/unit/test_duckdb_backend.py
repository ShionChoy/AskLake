# tests/unit/test_duckdb_backend.py
from engine.lakehouse.duckdb_backend import DuckDBBackend
from engine.ports.storage import StorageBackend

SEED_SQL = """
CREATE TABLE movies AS
SELECT * FROM (VALUES
    ('Inception', 2010, 8.8),
    ('Interstellar', 2014, 8.7),
    ('Tenet', 2020, 7.3)
) AS t(title, year, rating);
"""


def test_backend_satisfies_protocol():
    backend = DuckDBBackend()
    assert isinstance(backend, StorageBackend)


def test_run_sql_returns_columns_and_rows():
    backend = DuckDBBackend()
    backend.setup(SEED_SQL)
    sql = "SELECT title, rating FROM movies WHERE year >= 2014 ORDER BY rating DESC"
    result = backend.run_sql(sql)
    assert result.columns == ["title", "rating"]
    assert result.rows[0] == ("Interstellar", 8.7)
    assert len(result.rows) == 2


def test_list_tables_reports_columns():
    backend = DuckDBBackend()
    backend.setup(SEED_SQL)
    tables = backend.list_tables()
    names = {t.name for t in tables}
    assert "movies" in names
    movies = next(t for t in tables if t.name == "movies")
    colnames = {c.name for c in movies.columns}
    assert {"title", "year", "rating"} <= colnames


def test_parquet_dir_registers_views(tmp_path):
    # build a parquet file via a throwaway connection
    seed = DuckDBBackend()
    seed.setup(SEED_SQL)
    pq = tmp_path / "movies.parquet"
    seed.run_sql(f"COPY movies TO '{pq}' (FORMAT PARQUET)")
    backend = DuckDBBackend(parquet_dir=str(tmp_path))
    result = backend.run_sql("SELECT COUNT(*) AS n FROM movies")
    assert result.rows[0][0] == 3


def test_parquet_view_name_sanitizes_dots_and_hyphens(tmp_path):
    """Stems like 'title.basics' or 'my-table' must be sanitized to underscores.

    DuckDB interprets 'title.basics' as schema.table in unquoted DDL,
    and hyphens cause a parser error.  The backend must sanitize and quote
    the view name so the file is accessible as e.g. 'title_basics'.
    """
    seed = DuckDBBackend()
    seed.setup(SEED_SQL)
    # Use a dotted stem that mirrors the real IMDb file name title.basics.parquet
    pq = tmp_path / "title.basics.parquet"
    seed.run_sql(f"COPY movies TO '{pq}' (FORMAT PARQUET)")
    backend = DuckDBBackend(parquet_dir=str(tmp_path))
    # The sanitized view name must be queryable
    result = backend.run_sql("SELECT COUNT(*) AS n FROM title_basics")
    assert result.rows[0][0] == 3
