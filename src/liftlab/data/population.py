"""Synthetic Olist-shaped population generator.

When the real Olist dataset is unavailable (no Kaggle credentials), LiftLab
generates a *synthetic* population that mirrors the Olist schema exactly, same
five CSV files, same column names, so the downstream dbt models are byte-for-byte
identical regardless of source.

DISCLOSURE: this is a synthetic stand-in for the real Olist e-commerce population.
It is realistic enough to exercise the SQL layer and to host the synthetic
experiment, but it is *not* real customer data. The treatment effect injected on
top of any population (real or synthetic) is separately synthetic; see
``liftlab.simulation``.

Design notes
------------
* Each customer has latent traits (spend level + order frequency) that induce
  organic correlation between their pre-period and post-period behaviour, useful
  substrate for CUPED, though the experiment simulator controls the covariate
  correlation precisely in Phase 2.
* Generation is fully vectorized and seeded: identical seed -> identical CSV bytes.
* IDs are 32-char hex strings derived deterministically from integer indices to
  mimic Olist's hash-like identifiers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from liftlab.config import Config

# Brazilian states with rough Olist-like weights (SP dominates).
_STATES = np.array(["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "DF", "GO", "ES", "PE", "CE"])
_STATE_WEIGHTS = np.array(
    [0.42, 0.13, 0.12, 0.055, 0.05, 0.036, 0.034, 0.022, 0.02, 0.02, 0.018, 0.015]
)
_STATE_WEIGHTS = _STATE_WEIGHTS / _STATE_WEIGHTS.sum()

_PAYMENT_TYPES = np.array(["credit_card", "boleto", "voucher", "debit_card"])
_PAYMENT_WEIGHTS = np.array([0.74, 0.19, 0.05, 0.02])
_PAYMENT_WEIGHTS = _PAYMENT_WEIGHTS / _PAYMENT_WEIGHTS.sum()

# Olist review scores are heavily skewed to 5/4.
_REVIEW_SCORES = np.array([1, 2, 3, 4, 5])
_REVIEW_WEIGHTS = np.array([0.11, 0.03, 0.08, 0.19, 0.59])
_REVIEW_WEIGHTS = _REVIEW_WEIGHTS / _REVIEW_WEIGHTS.sum()

_ORDER_STATUSES = np.array(["delivered", "shipped", "canceled", "invoiced"])
_STATUS_WEIGHTS = np.array([0.97, 0.015, 0.01, 0.005])
_STATUS_WEIGHTS = _STATUS_WEIGHTS / _STATUS_WEIGHTS.sum()


def _hex_ids(prefix_salt: int, n: int) -> np.ndarray:
    """Deterministic 32-char hex-ish IDs from a salted integer sequence."""
    idx = np.arange(n, dtype=np.uint64)
    # Mix the index with a salt so different entity types don't collide visually.
    mixed = (idx + np.uint64(prefix_salt) * np.uint64(2654435761)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return np.array(
        [f"{int(v):016x}{(int(v) ^ prefix_salt) & 0xFFFFFFFFFFFFFFFF:016x}" for v in mixed]
    )


def generate_population(config: Config, n_customers: int | None = None) -> dict[str, pd.DataFrame]:
    """Generate the five Olist-shaped tables as pandas DataFrames.

    Parameters
    ----------
    config:
        Experiment config (provides seed + synthetic date range).
    n_customers:
        Override the configured customer count (used by fast tests).
    """
    syn = config.data.synthetic
    n_cust = int(n_customers if n_customers is not None else syn.n_customers)
    rng = np.random.default_rng(config.seed)

    start = pd.Timestamp(syn.start_date)
    end = pd.Timestamp(syn.end_date)
    span_seconds = int((end - start).total_seconds())

    # --- Customer latent traits ------------------------------------------------
    # Spend level (BRL) and order frequency. Heavy right tails like real spend.
    latent_spend = rng.lognormal(mean=4.6, sigma=0.6, size=n_cust)  # ~ median 100 BRL
    latent_freq = rng.gamma(shape=1.4, scale=0.65, size=n_cust)
    n_orders_per_cust = 1 + rng.poisson(latent_freq)  # >= 1 order per customer

    customer_unique_ids = _hex_ids(1, n_cust)
    cust_states = rng.choice(_STATES, size=n_cust, p=_STATE_WEIGHTS)
    cust_zip = rng.integers(1000, 99999, size=n_cust)
    cust_city = np.char.add("city_", cust_states.astype(str))

    # --- Expand to orders ------------------------------------------------------
    total_orders = int(n_orders_per_cust.sum())
    cust_of_order = np.repeat(np.arange(n_cust), n_orders_per_cust)

    order_ids = _hex_ids(2, total_orders)
    customer_ids = _hex_ids(3, total_orders)  # order-scoped, one per order (Olist style)

    purchase_offsets = rng.integers(0, span_seconds, size=total_orders)
    purchase_ts = start + pd.to_timedelta(purchase_offsets, unit="s")

    statuses = rng.choice(_ORDER_STATUSES, size=total_orders, p=_STATUS_WEIGHTS)
    approved_ts = purchase_ts + pd.to_timedelta(rng.integers(1, 48, total_orders), unit="h")
    carrier_ts = purchase_ts + pd.to_timedelta(rng.integers(1, 5, total_orders), unit="D")
    delivered_ts = purchase_ts + pd.to_timedelta(rng.integers(3, 15, total_orders), unit="D")
    estimated_ts = purchase_ts + pd.to_timedelta(rng.integers(7, 25, total_orders), unit="D")

    # --- Order items (1-3 per order) ------------------------------------------
    n_items = 1 + rng.poisson(0.4, size=total_orders)
    n_items = np.clip(n_items, 1, 5)
    total_items = int(n_items.sum())
    item_order_idx = np.repeat(np.arange(total_orders), n_items)

    # Per-order spend driven by the customer's latent spend level + noise.
    order_value = latent_spend[cust_of_order] * rng.lognormal(0.0, 0.35, total_orders)
    # Split each order's value across its items (roughly equal + noise).
    item_share_noise = rng.uniform(0.7, 1.3, total_items)
    items_per_order = n_items[item_order_idx]
    item_price = (order_value[item_order_idx] / items_per_order) * item_share_noise
    item_price = np.round(item_price, 2)
    freight_value = np.round(item_price * rng.uniform(0.05, 0.25, total_items), 2)

    product_ids = _hex_ids(4, total_items % 5000 + 5000)  # small product pool
    seller_ids = _hex_ids(5, total_items % 800 + 800)
    product_pick = rng.integers(0, len(product_ids), total_items)
    seller_pick = rng.integers(0, len(seller_ids), total_items)
    shipping_limit = purchase_ts[item_order_idx] + pd.to_timedelta(
        rng.integers(2, 8, total_items), unit="D"
    )
    # order_item_id is 1..k within each order.
    order_item_id = _within_group_rank(item_order_idx)

    # --- Payments (one per order) ---------------------------------------------
    order_total_value = (
        pd.Series(item_price + freight_value).groupby(item_order_idx).sum().to_numpy()
    )
    payment_types = rng.choice(_PAYMENT_TYPES, size=total_orders, p=_PAYMENT_WEIGHTS)
    installments = rng.integers(1, 11, size=total_orders)

    # --- Reviews (~90% of orders) ---------------------------------------------
    has_review = rng.random(total_orders) < 0.90
    review_idx = np.flatnonzero(has_review)
    review_scores = rng.choice(_REVIEW_SCORES, size=review_idx.size, p=_REVIEW_WEIGHTS)
    review_ids = _hex_ids(6, review_idx.size)
    review_creation = purchase_ts[review_idx] + pd.to_timedelta(
        rng.integers(1, 10, review_idx.size), unit="D"
    )
    review_answer = review_creation + pd.to_timedelta(rng.integers(0, 5, review_idx.size), unit="D")

    fmt = "%Y-%m-%d %H:%M:%S"

    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "customer_unique_id": customer_unique_ids[cust_of_order],
            "customer_zip_code_prefix": cust_zip[cust_of_order],
            "customer_city": cust_city[cust_of_order],
            "customer_state": cust_states[cust_of_order],
        }
    )

    orders = pd.DataFrame(
        {
            "order_id": order_ids,
            "customer_id": customer_ids,
            "order_status": statuses,
            "order_purchase_timestamp": purchase_ts.strftime(fmt),
            "order_approved_at": approved_ts.strftime(fmt),
            "order_delivered_carrier_date": carrier_ts.strftime(fmt),
            "order_delivered_customer_date": delivered_ts.strftime(fmt),
            "order_estimated_delivery_date": estimated_ts.strftime(fmt),
        }
    )

    order_items = pd.DataFrame(
        {
            "order_id": order_ids[item_order_idx],
            "order_item_id": order_item_id,
            "product_id": product_ids[product_pick],
            "seller_id": seller_ids[seller_pick],
            "shipping_limit_date": shipping_limit.strftime(fmt),
            "price": item_price,
            "freight_value": freight_value,
        }
    )

    order_payments = pd.DataFrame(
        {
            "order_id": order_ids,
            "payment_sequential": np.ones(total_orders, dtype=int),
            "payment_type": payment_types,
            "payment_installments": installments,
            "payment_value": np.round(order_total_value, 2),
        }
    )

    order_reviews = pd.DataFrame(
        {
            "review_id": review_ids,
            "order_id": order_ids[review_idx],
            "review_score": review_scores,
            "review_comment_title": "",
            "review_comment_message": "",
            "review_creation_date": review_creation.strftime(fmt),
            "review_answer_timestamp": review_answer.strftime(fmt),
        }
    )

    return {
        "olist_customers_dataset": customers,
        "olist_orders_dataset": orders,
        "olist_order_items_dataset": order_items,
        "olist_order_payments_dataset": order_payments,
        "olist_order_reviews_dataset": order_reviews,
    }


def _within_group_rank(group_ids: np.ndarray) -> np.ndarray:
    """1-based rank within each contiguous group (group_ids is sorted/repeated).

    Fully vectorized: for each position, find where its group started and take the
    offset. ``group_ids`` is assumed non-decreasing (it is ``np.repeat(arange, k)``).
    """
    n = group_ids.size
    if n == 0:
        return np.ones(0, dtype=int)
    positions = np.arange(n)
    boundaries = np.flatnonzero(np.diff(group_ids) != 0) + 1  # first index of each new group
    group_start = np.zeros(n, dtype=int)
    group_start[boundaries] = boundaries
    np.maximum.accumulate(group_start, out=group_start)
    return positions - group_start + 1


def write_population_csvs(
    config: Config, out_dir: Path, n_customers: int | None = None
) -> dict[str, int]:
    """Generate and write the five Olist-shaped CSVs. Returns row counts per table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = generate_population(config, n_customers=n_customers)
    counts: dict[str, int] = {}
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
        counts[name] = len(df)
    return counts
