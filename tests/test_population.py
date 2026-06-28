"""Synthetic population: correct schema, non-empty, and deterministic."""

from __future__ import annotations

import hashlib
from pathlib import Path

from liftlab.config import Config
from liftlab.data.population import generate_population, write_population_csvs
from liftlab.paths import OLIST_TABLES

EXPECTED_COLUMNS = {
    "olist_customers_dataset": [
        "customer_id",
        "customer_unique_id",
        "customer_zip_code_prefix",
        "customer_city",
        "customer_state",
    ],
    "olist_orders_dataset": [
        "order_id",
        "customer_id",
        "order_status",
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "olist_order_items_dataset": [
        "order_id",
        "order_item_id",
        "product_id",
        "seller_id",
        "shipping_limit_date",
        "price",
        "freight_value",
    ],
    "olist_order_payments_dataset": [
        "order_id",
        "payment_sequential",
        "payment_type",
        "payment_installments",
        "payment_value",
    ],
    "olist_order_reviews_dataset": [
        "review_id",
        "order_id",
        "review_score",
        "review_comment_title",
        "review_comment_message",
        "review_creation_date",
        "review_answer_timestamp",
    ],
}


def test_population_has_olist_schema(small_config: Config) -> None:
    tables = generate_population(small_config, n_customers=300)
    assert set(tables) == set(OLIST_TABLES)
    for name, df in tables.items():
        assert list(df.columns) == EXPECTED_COLUMNS[name], name
        assert len(df) > 0, name


def test_population_is_internally_consistent(small_config: Config) -> None:
    tables = generate_population(small_config, n_customers=300)
    customers = tables["olist_customers_dataset"]
    orders = tables["olist_orders_dataset"]
    items = tables["olist_order_items_dataset"]
    # One customer_id row per order; every order's customer_id is known.
    assert len(customers) == len(orders)
    assert customers["customer_id"].is_unique
    assert orders["customer_id"].isin(set(customers["customer_id"])).all()
    # Every line item belongs to a real order; prices are non-negative.
    assert items["order_id"].isin(set(orders["order_id"])).all()
    assert (items["price"] >= 0).all()
    assert (items["freight_value"] >= 0).all()


def test_population_is_deterministic(small_config: Config, tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    write_population_csvs(small_config, a, n_customers=300)
    write_population_csvs(small_config, b, n_customers=300)
    for name in OLIST_TABLES:
        ha = hashlib.sha256((a / f"{name}.csv").read_bytes()).hexdigest()
        hb = hashlib.sha256((b / f"{name}.csv").read_bytes()).hexdigest()
        assert ha == hb, f"{name} not reproducible under a fixed seed"
