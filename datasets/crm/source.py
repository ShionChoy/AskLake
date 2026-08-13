"""Synthetic CRM connector used to verify cross-dataset generalization.

Deterministic — no randomness — so the
gold set is stable. The engine never imports this; it is dataset-specific config, exactly like
datasets/imdb/source.py. Output parquet is gitignored and regenerable via `make build-crm`."""

from __future__ import annotations

from pathlib import Path

import duckdb


def build_parquet(out_dir: str, n_customers: int = 48) -> list[str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    con.execute(
        """
        CREATE TABLE regions AS SELECT * FROM (VALUES
            ('APAC', 'Singapore'), ('EMEA', 'Germany'),
            ('AMER', 'United States'), ('LATAM', 'Brazil')
        ) t(region, country)
        """
    )
    con.execute(
        f"""
        CREATE TABLE customers AS
        SELECT
            i AS customer_id,
            'Customer ' || CAST(i AS VARCHAR) AS name,
            CASE (i % 4) WHEN 0 THEN 'APAC' WHEN 1 THEN 'EMEA' WHEN 2 THEN 'AMER' ELSE 'LATAM' END
                AS region,
            2018 + (i % 6) AS signup_year,
            CASE WHEN i % 5 = 0 THEN 'Churned' ELSE 'Active' END AS status
        FROM range(1, {int(n_customers) + 1}) AS t(i)
        """
    )
    con.execute(
        """
        CREATE TABLE subscriptions AS
        SELECT
            customer_id AS subscription_id,
            customer_id,
            CASE (customer_id % 3) WHEN 0 THEN 'Enterprise' WHEN 1 THEN 'Basic' ELSE 'Pro' END
                AS plan,
            CASE (customer_id % 3) WHEN 0 THEN 100 WHEN 1 THEN 10 ELSE 30 END AS monthly_fee,
            (status = 'Active') AS active
        FROM customers
        """
    )
    con.execute(
        """
        CREATE TABLE support_tickets AS
        SELECT
            row_number() OVER () AS ticket_id,
            c.customer_id,
            CASE (c.customer_id % 3) WHEN 0 THEN 'high' WHEN 1 THEN 'low' ELSE 'medium' END
                AS priority,
            (c.customer_id % 2 = 0) AS resolved
        FROM customers c
        JOIN range(1, 4) AS g(k) ON k <= (c.customer_id % 3) + 1
        """
    )

    written: list[str] = []
    for tbl in ("customers", "subscriptions", "support_tickets", "regions"):
        path = (out / f"{tbl}.parquet").as_posix()
        con.execute(f"COPY {tbl} TO '{path}' (FORMAT PARQUET)")
        written.append(path)
    return written
