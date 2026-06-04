from engine.lakehouse.duckdb_backend import DuckDBBackend
from eval.harness import EvalCase, evaluate, score_case

SCHEMA = "CREATE TABLE t AS SELECT * FROM (VALUES (1), (2), (3)) v(x);"


def _backend():
    b = DuckDBBackend()
    b.setup(SCHEMA)
    return b


def test_score_case_correct_and_valid():
    valid, correct = score_case("SELECT x FROM t ORDER BY x", "SELECT x FROM t", _backend())
    assert valid and correct  # multiset match, order-insensitive


def test_score_case_invalid_sql():
    valid, correct = score_case("SELECT nope FROM t", "SELECT x FROM t", _backend())
    assert not valid and not correct


def test_score_case_valid_but_wrong():
    valid, correct = score_case("SELECT x FROM t WHERE x > 2", "SELECT x FROM t", _backend())
    assert valid and not correct


def test_evaluate_aggregates():
    cases = [EvalCase("c1", SCHEMA, "all x", "SELECT x FROM t")]
    report = evaluate("sys", cases, lambda case, backend: ("SELECT x FROM t ORDER BY x", 0))
    assert report.n == 1
    assert report.execution_accuracy == 1.0
    assert report.valid_sql_rate == 1.0
    assert report.avg_attempts == 0.0
