"""Raw CSVs load into DuckDB and are queryable."""

from __future__ import annotations

from pathlib import Path

import duckdb

from liftlab.paths import OLIST_TABLES


def test_raw_tables_loaded(warehouse_db: Path) -> None:
    con = duckdb.connect(str(warehouse_db), read_only=True)
    try:
        schemas = {
            r[0]
            for r in con.execute("SELECT schema_name FROM information_schema.schemata").fetchall()
        }
        assert "raw" in schemas
        for name in OLIST_TABLES:
            n = con.execute(f"SELECT count(*) FROM raw.{name}").fetchone()[0]
            assert n > 0, name
    finally:
        con.close()


def test_raw_orders_join_customers(warehouse_db: Path) -> None:
    con = duckdb.connect(str(warehouse_db), read_only=True)
    try:
        matched = con.execute(
            """
            SELECT count(*)
            FROM raw.olist_orders_dataset o
            JOIN raw.olist_customers_dataset c USING (customer_id)
            """
        ).fetchone()[0]
        total_orders = con.execute("SELECT count(*) FROM raw.olist_orders_dataset").fetchone()[0]
        assert matched == total_orders
    finally:
        con.close()
