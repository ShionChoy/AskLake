"""Hand-authored NL -> gold SQL over the synthetic CRM parquet (built by `make build-crm` into
PARQUET_DIR). Mirrors the IMDb gold tiers (aggregation / topn / multihop) so the same ablation
runs on a dataset the engine was never tuned for. schema_sql is unused (runs against the shared
built backend), matching eval/imdb_gold.py."""

from __future__ import annotations

from eval.harness import EvalCase

PARQUET_DIR = "data/crm/parquet"


def _c(name, question, gold_sql, tier):
    return EvalCase(name=name, schema_sql="", question=question, gold_sql=gold_sql, tier=tier)


CRM_GOLD: list[EvalCase] = [
    # --- aggregation (no LIMIT; unconditionally tie-safe) ---
    _c(
        "count_churned",
        "How many churned customers are there?",
        "SELECT count(*) FROM customers WHERE status = 'Churned'",
        "aggregation",
    ),
    _c(
        "count_by_region",
        "How many customers are in each region?",
        "SELECT region, count(*) FROM customers GROUP BY region",
        "aggregation",
    ),
    _c(
        "avg_fee_by_plan",
        "What is the average monthly fee per plan?",
        "SELECT plan, avg(monthly_fee) FROM subscriptions GROUP BY plan",
        "aggregation",
    ),
    _c(
        "count_active",
        "How many active customers are there?",
        "SELECT count(*) FROM customers WHERE status = 'Active'",
        "aggregation",
    ),
    # --- topn (LIMIT; plan revenue is strictly separated: 1600 > 480 > 160) ---
    _c(
        "top2_plans_by_revenue",
        "Which 2 plans bring the most total monthly fee?",
        "SELECT plan, sum(monthly_fee) AS rev FROM subscriptions GROUP BY plan "
        "ORDER BY rev DESC LIMIT 2",
        "topn",
    ),
    _c(
        "top_plan_by_revenue",
        "Which plan brings in the most total monthly fee?",
        "SELECT plan, sum(monthly_fee) AS rev FROM subscriptions GROUP BY plan "
        "ORDER BY rev DESC LIMIT 1",
        "topn",
    ),
    # --- multihop (joins; categorical value-linking on region/status/plan/priority) ---
    _c(
        "apac_total_fee",
        "What is the total monthly fee from customers in APAC?",
        "SELECT sum(s.monthly_fee) FROM subscriptions s JOIN customers c USING (customer_id) "
        "WHERE c.region = 'APAC'",
        "multihop",
    ),
    _c(
        "churned_enterprise",
        "How many Enterprise customers have churned?",
        "SELECT count(*) FROM subscriptions s JOIN customers c USING (customer_id) "
        "WHERE s.plan = 'Enterprise' AND c.status = 'Churned'",
        "multihop",
    ),
    _c(
        "region_most_churn",
        "Which region has the most churned customers?",
        "SELECT region, count(*) AS n FROM customers WHERE status = 'Churned' "
        "GROUP BY region ORDER BY n DESC, region LIMIT 1",
        "multihop",
    ),
    _c(
        "high_priority_active",
        "How many high-priority tickets come from active customers?",
        "SELECT count(*) FROM support_tickets t JOIN customers c USING (customer_id) "
        "WHERE t.priority = 'high' AND c.status = 'Active'",
        "multihop",
    ),
    _c(
        "avg_fee_apac",
        "What is the average monthly fee for APAC customers?",
        "SELECT avg(s.monthly_fee) FROM subscriptions s JOIN customers c USING (customer_id) "
        "WHERE c.region = 'APAC'",
        "multihop",
    ),
]
