from __future__ import annotations

from pathlib import Path

import pytest

from engine.lakehouse.duckdb_backend import DuckDBBackend
from eval.imdb_gold import IMDB_GOLD, PARQUET_DIR

_ROOT = Path(__file__).resolve().parents[2]
_PARQUET = _ROOT / PARQUET_DIR


def test_gold_set_shape():
    assert len(IMDB_GOLD) >= 12
    names = [c.name for c in IMDB_GOLD]
    assert len(names) == len(set(names))  # unique names
    for c in IMDB_GOLD:
        assert c.question.strip() and c.gold_sql.strip()


@pytest.mark.skipif(not _PARQUET.exists(), reason="IMDb parquet not built (run make build-imdb)")
def test_every_gold_sql_executes_and_returns_rows():
    backend = DuckDBBackend(parquet_dir=str(_PARQUET))
    for c in IMDB_GOLD:
        res = backend.run_sql(c.gold_sql)
        assert res.rows, f"gold SQL returned no rows for {c.name}: {c.gold_sql}"
