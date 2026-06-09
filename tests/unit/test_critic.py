from engine.agents.critic import Critique, assess, select_consistent
from engine.ports.storage import QueryResult


def _qr(rows):
    return QueryResult(columns=["title", "r"], rows=rows)


def test_assess_accepts_good_topn_result():
    c = assess(
        "top 5 highest rated movies",
        "SELECT title, r FROM m ORDER BY r DESC LIMIT 5",
        _qr([("A", 9.0)]),
        difficulty="hard",
    )
    assert c == Critique(ok=True, reasons=())


def test_assess_flags_topn_without_order_or_limit():
    c = assess(
        "top 5 highest rated movies", "SELECT title, r FROM m", _qr([("A", 9.0)]), difficulty="hard"
    )
    assert c.ok is False
    assert "ORDER BY" in " ".join(c.reasons)
    assert "LIMIT" in " ".join(c.reasons)


def test_assess_flags_zero_rows():
    c = assess(
        "most popular movies",
        "SELECT title, r FROM m WHERE 1=0 LIMIT 5 ORDER BY r DESC",
        _qr([]),
        difficulty="hard",
    )
    assert c.ok is False and "0 rows" in " ".join(c.reasons)


def test_assess_flags_no_result():
    c = assess("q", "SELECT 1", None, difficulty="simple")
    assert c.ok is False


def test_select_consistent_majority_vote_discards_minority():
    good = _qr([("A", 9.0), ("B", 8.0)])
    bad = _qr([("Z", 1.0)])
    chosen_sql, chosen_res = select_consistent(
        [("sql_bad", bad), ("sql_good_1", good), ("sql_good_2", good)]
    )
    assert chosen_res.rows == good.rows
    assert chosen_sql == "sql_good_1"  # first member of the winning group


def test_select_consistent_is_order_insensitive():
    a = _qr([("A", 9.0), ("B", 8.0)])
    b = _qr([("B", 8.0), ("A", 9.0)])  # same multiset, different order
    _sql, res = select_consistent([("s1", a), ("s2", b)])
    assert res.rows in (a.rows, b.rows)  # treated as one group of 2
