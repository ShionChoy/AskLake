from __future__ import annotations

import re
from pathlib import Path

import pytest

from engine.lakehouse.duckdb_backend import DuckDBBackend
from eval.imdb_gold import (
    _MULTIHOP_PROBES,
    _TOPN_PARAMS,
    IMDB_GOLD,
    PARQUET_DIR,
    _topn_probe_sql,
)

_ROOT = Path(__file__).resolve().parents[2]
_PARQUET = _ROOT / PARQUET_DIR


def _normalize_sql(s: str) -> str:
    """Collapse all whitespace runs to a single space for duplicate detection."""
    return re.sub(r"\s+", " ", s).strip()


def test_gold_set_shape():
    assert len(IMDB_GOLD) >= 95
    names = [c.name for c in IMDB_GOLD]
    assert len(names) == len(set(names))  # unique names
    for c in IMDB_GOLD:
        assert c.question.strip() and c.gold_sql.strip()
        assert c.tier in {"aggregation", "topn", "multihop"}
    tiers = {c.tier for c in IMDB_GOLD}
    assert tiers == {"aggregation", "topn", "multihop"}  # all three represented


def test_no_duplicate_questions_or_sql():
    """Prevent I-1 class of bug: duplicated question or gold_sql across cases."""
    questions = [c.question for c in IMDB_GOLD]
    assert len(questions) == len(set(questions)), "Duplicate question found in IMDB_GOLD: " + str(
        [q for q in questions if questions.count(q) > 1]
    )
    sqls = [_normalize_sql(c.gold_sql) for c in IMDB_GOLD]
    assert len(sqls) == len(set(sqls)), "Duplicate gold_sql found in IMDB_GOLD: " + str(
        [s for s in sqls if sqls.count(s) > 1][:1]
    )


@pytest.mark.skipif(not _PARQUET.exists(), reason="IMDb parquet not built (run make build-imdb)")
def test_every_gold_sql_executes_and_returns_rows():
    backend = DuckDBBackend(parquet_dir=str(_PARQUET))
    for c in IMDB_GOLD:
        res = backend.run_sql(c.gold_sql)
        assert res.rows, f"gold SQL returned no rows for {c.name}: {c.gold_sql}"


@pytest.mark.skipif(not _PARQUET.exists(), reason="IMDb parquet not built (run make build-imdb)")
def test_topn_cases_are_tie_safe():
    """No boundary tie may span a LIMIT N cut, else multiset-exact scoring is ambiguous."""
    backend = DuckDBBackend(parquet_dir=str(_PARQUET))
    for p in _TOPN_PARAMS:
        slug, _q, _where, _key, _proj, n = p
        rows = backend.run_sql(_topn_probe_sql(p)).rows
        if len(rows) > n:  # a cut exists only when more than N rows qualify
            assert rows[n - 1][0] > rows[n][0], f"boundary tie in topn case {slug}"


@pytest.mark.skipif(not _PARQUET.exists(), reason="IMDb parquet not built (run make build-imdb)")
def test_multihop_limit_cases_are_tie_safe():
    """All multihop LIMIT N cases must be strictly separated at the boundary."""
    backend = DuckDBBackend(parquet_dir=str(_PARQUET))
    for name, probe_sql, n in _MULTIHOP_PROBES:
        rows = backend.run_sql(probe_sql).rows
        if len(rows) > n:  # a cut exists only when more than N rows qualify
            assert rows[n - 1][0] > rows[n][0], f"boundary tie in multihop case {name}"


def test_aggregation_golds_have_no_gratuitous_round():
    from eval.imdb_gold import IMDB_GOLD

    offenders = [
        c.name for c in IMDB_GOLD if c.tier == "aggregation" and "ROUND(" in c.gold_sql.upper()
    ]
    assert offenders == [], (
        f"aggregation golds should return canonical (unrounded) aggregates: {offenders}"
    )


def test_no_rounding_decade_binning():
    """Decade binning must floor, not round (CAST(double AS INT) rounds in DuckDB, leaking
    years across decade boundaries). Correct form is integer floor-division (// )."""
    from eval.imdb_gold import IMDB_GOLD

    offenders = [c.name for c in IMDB_GOLD if "/10 AS INT)*10" in c.gold_sql]
    assert offenders == [], f"buggy rounding decade binning still present in: {offenders}"
