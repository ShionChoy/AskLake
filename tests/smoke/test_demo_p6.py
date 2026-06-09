from demos.demo_p6 import run_demo_p6


def test_demo_p6_grounds_and_self_consistently_answers():
    out = run_demo_p6()
    assert out["rows"][0][0] == "Inception"  # top-rated row, grounded + correct
    assert "LIMIT" in out["sql"].upper()
    assert out["discarded_minority"] is True  # a wrong candidate was voted out
    assert out["corrections"] == 0  # critic accepted the consistent answer
