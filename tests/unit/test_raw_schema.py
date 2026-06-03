from engine.lakehouse.duckdb_backend import DuckDBBackend
from engine.ports.schema import SchemaProvider
from engine.semantic.raw_schema import RawSchemaProvider


def test_raw_schema_lists_tables_and_columns():
    backend = DuckDBBackend()
    backend.setup("CREATE TABLE t AS SELECT 1::INTEGER AS a, 'x' AS b;")
    sp = RawSchemaProvider(backend)
    assert isinstance(sp, SchemaProvider)
    ctx = sp.schema_context("anything")
    assert "t(" in ctx
    assert "a" in ctx and "b" in ctx
