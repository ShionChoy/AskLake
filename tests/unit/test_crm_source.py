from datasets.crm.source import build_parquet
from engine.lakehouse.duckdb_backend import DuckDBBackend


def test_crm_build_is_deterministic_and_well_formed(tmp_path):
    out = tmp_path / "crm"
    files = build_parquet(str(out))
    assert len(files) == 4  # customers, subscriptions, support_tickets, regions
    b = DuckDBBackend(parquet_dir=str(out))
    names = {t.name for t in b.list_tables()}
    assert {"customers", "subscriptions", "support_tickets", "regions"} <= names
    n1 = b.run_sql("SELECT count(*) FROM customers").rows[0][0]
    build_parquet(str(out))
    n2 = DuckDBBackend(parquet_dir=str(out)).run_sql("SELECT count(*) FROM customers").rows[0][0]
    assert n1 == n2 and n1 >= 20
    statuses = {r[0] for r in b.run_sql("SELECT DISTINCT status FROM customers").rows}
    assert {"Active", "Churned"} <= statuses
    regions = {r[0] for r in b.run_sql("SELECT DISTINCT region FROM regions").rows}
    assert {"APAC", "EMEA", "AMER", "LATAM"} <= regions
