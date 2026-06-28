"""Canonical filesystem locations for the project.

The package is installed editable (``uv sync``), so ``__file__`` lives inside the
repo. We walk up to the directory that contains ``pyproject.toml`` to find the
project root, which works both locally and inside the Docker image (WORKDIR=/app).
An explicit ``LIFTLAB_ROOT`` env var overrides the search if ever needed.
"""

from __future__ import annotations

import os
from pathlib import Path


def _find_project_root() -> Path:
    env_root = os.environ.get("LIFTLAB_ROOT")
    if env_root:
        return Path(env_root).resolve()
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    # Fallback: current working directory.
    return Path.cwd().resolve()


PROJECT_ROOT: Path = _find_project_root()

CONFIG_PATH: Path = PROJECT_ROOT / "config" / "experiment.yaml"

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
WAREHOUSE_DIR: Path = DATA_DIR / "warehouse"

DBT_DIR: Path = PROJECT_ROOT / "dbt"
RUNS_DIR: Path = PROJECT_ROOT / "runs"
DOCS_DIR: Path = PROJECT_ROOT / "docs"


def duckdb_path() -> Path:
    """Path to the DuckDB warehouse file (overridable via ``LIFTLAB_DUCKDB``)."""
    env = os.environ.get("LIFTLAB_DUCKDB")
    if env:
        return Path(env).resolve()
    return WAREHOUSE_DIR / "liftlab.duckdb"


# The five Olist CSV file stems (== raw table names). The synthetic generator
# emits exactly these so the dbt models are identical for real or synthetic data.
OLIST_TABLES: tuple[str, ...] = (
    "olist_customers_dataset",
    "olist_orders_dataset",
    "olist_order_items_dataset",
    "olist_order_payments_dataset",
    "olist_order_reviews_dataset",
)
