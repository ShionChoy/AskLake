from demos.demo_p7 import run_demo_p7


def test_demo_p7_roles_differ():
    out = run_demo_p7()
    # public sees fewer rows than analyst (row-level security) and masked birth years
    assert out["analyst_rows"] > out["public_rows"]
    assert out["public_birthyear_masked"] is True
    assert out["blocked_without_limit"] is True
