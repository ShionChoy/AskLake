from demos.demo_p3 import run_demo_p3


def test_demo_p3_rbac_pii_and_cost_guardrail():
    out = run_demo_p3()
    # analyst sees full, unmasked data (both rows, incl. the restricted one)
    assert out["analyst"]["rows"] == [["Nolan", 1970, "movie"], ["Hidden", 1980, "adult"]]
    # public: restricted row filtered out + PII (birthYear) masked
    assert out["public"]["rows"] == [["Nolan", "***", "movie"]]
    # a query without LIMIT is intercepted by the cost guardrail
    assert out["cost_guardrail_block"] and "guardrail" in out["cost_guardrail_block"].lower()
