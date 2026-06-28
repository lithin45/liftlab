"""The dbt project builds the experiment fact table, and it is queryable.

This is the Phase 1 acceptance gate: dbt builds staging + the customer_metrics
mart (running dbt's own not_null/unique/accepted_values + the singular invariant
test), and a DuckDB query against the mart returns sensible aggregates.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import duckdb
import pandas as pd
import pytest

pytestmark = pytest.mark.slow


def test_customer_metrics_built_and_valid(dbt_built: Path) -> None:
    con = duckdb.connect(str(dbt_built), read_only=True)
    try:
        # The mart exists and is non-empty.
        n = con.execute("SELECT count(*) FROM main.customer_metrics").fetchone()[0]
        assert n > 0

        # Grain integrity: customer_unique_id is unique.
        dupes = con.execute(
            """
            SELECT count(*) FROM (
                SELECT customer_unique_id
                FROM main.customer_metrics
                GROUP BY customer_unique_id
                HAVING count(*) > 1
            )
            """
        ).fetchone()[0]
        assert dupes == 0

        # Invariants the analysis relies on.
        bad = con.execute(
            """
            SELECT count(*) FROM main.customer_metrics
            WHERE n_orders < 1
               OR total_value < 0
               OR (n_orders_pre + n_orders_post) <> n_orders
               OR post_converted NOT IN (0, 1)
            """
        ).fetchone()[0]
        assert bad == 0
    finally:
        con.close()


def test_pre_and_post_periods_are_populated(dbt_built: Path) -> None:
    """The cutoff split must yield both pre- and post-period activity (CUPED needs it)."""
    con = duckdb.connect(str(dbt_built), read_only=True)
    try:
        pre, post = con.execute(
            """
            SELECT sum(n_orders_pre), sum(n_orders_post)
            FROM main.customer_metrics
            """
        ).fetchone()
        assert pre > 0
        assert post > 0
    finally:
        con.close()


def test_pre_post_split_matches_independent_recompute(dbt_built: Path, small_config) -> None:
    """Discriminating check: the mart's pre/post counts must equal an independent
    recomputation of the cutoff split from staging (a wrong cutoff would fail this)."""
    cutoff = small_config.data.cutoff_date
    con = duckdb.connect(str(dbt_built), read_only=True)
    try:
        mismatches = con.execute(
            f"""
            WITH independent AS (
                SELECT
                    c.customer_unique_id,
                    sum(CASE WHEN o.order_purchase_timestamp < TIMESTAMP '{cutoff}'
                             THEN 1 ELSE 0 END) AS pre,
                    sum(CASE WHEN o.order_purchase_timestamp >= TIMESTAMP '{cutoff}'
                             THEN 1 ELSE 0 END) AS post
                FROM main.stg_orders o
                JOIN main.stg_customers c USING (customer_id)
                GROUP BY c.customer_unique_id
            )
            SELECT count(*)
            FROM main.customer_metrics m
            JOIN independent i USING (customer_unique_id)
            WHERE m.n_orders_pre <> i.pre OR m.n_orders_post <> i.post
            """
        ).fetchone()[0]
        assert mismatches == 0
    finally:
        con.close()


@pytest.fixture
def orphan_built(small_config, raw_csvs: Path, tmp_path: Path) -> Path:
    """A warehouse whose raw orders include one order with no matching customer."""
    from liftlab.data.build import run_dbt
    from liftlab.data.warehouse import load_raw_to_duckdb

    rdir = tmp_path / "raw"
    shutil.copytree(raw_csvs, rdir)
    orders_path = rdir / "olist_orders_dataset.csv"
    orders = pd.read_csv(orders_path, dtype=str)
    orphan = orders.iloc[[0]].copy()
    orphan["order_id"] = "orphan_order_xyz"
    orphan["customer_id"] = "customer_not_in_customers_table"
    pd.concat([orders, orphan], ignore_index=True).to_csv(orders_path, index=False)

    db_path = tmp_path / "wh.duckdb"
    load_raw_to_duckdb(db_path, rdir)
    run_dbt("build", small_config, db_path)
    return db_path


def test_orphan_orders_are_dropped_by_the_mart(orphan_built: Path) -> None:
    """Referential integrity: an order with no customer must not reach customer_metrics
    (a real-Olist property the clean synthetic data can't otherwise exercise)."""
    con = duckdb.connect(str(orphan_built), read_only=True)
    try:
        raw_orders = con.execute("SELECT count(*) FROM raw.olist_orders_dataset").fetchone()[0]
        joinable = con.execute(
            "SELECT count(*) FROM main.stg_orders o JOIN main.stg_customers c USING (customer_id)"
        ).fetchone()[0]
        mart_orders = con.execute("SELECT sum(n_orders) FROM main.customer_metrics").fetchone()[0]
        assert mart_orders == joinable
        assert raw_orders == joinable + 1  # the orphan is in raw but excluded from the mart
    finally:
        con.close()
