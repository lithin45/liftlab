"""Shared pytest fixtures.

The data-layer tests run the *real* dbt project against a *small* synthetic
population in temp directories, so they exercise the genuine SQL pipeline while
staying fast. Heavy fixtures are session-scoped and built once.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from liftlab.config import Config, from_dict
from liftlab.paths import CONFIG_PATH

# Small enough that dbt builds in a couple of seconds.
SMALL_N_CUSTOMERS = 600


@pytest.fixture(scope="session")
def small_config() -> Config:
    """The real config, forced to a small synthetic population for fast tests."""
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw["data"]["source"] = "synthetic"
    raw["data"]["synthetic"]["n_customers"] = SMALL_N_CUSTOMERS
    return from_dict(raw)


@pytest.fixture(scope="session")
def raw_csvs(small_config: Config, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate the five Olist-shaped CSVs into a temp dir once per session."""
    from liftlab.data.population import write_population_csvs

    out = tmp_path_factory.mktemp("raw")
    write_population_csvs(small_config, out)
    return out


@pytest.fixture(scope="session")
def warehouse_db(raw_csvs: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Load the raw CSVs into a temp DuckDB warehouse once per session."""
    from liftlab.data.warehouse import load_raw_to_duckdb

    db_path = tmp_path_factory.mktemp("warehouse") / "test.duckdb"
    load_raw_to_duckdb(db_path, raw_csvs)
    return db_path


@pytest.fixture(scope="session")
def dbt_built(warehouse_db: Path, small_config: Config) -> Path:
    """Run `dbt build` against the temp warehouse once per session."""
    from liftlab.data.build import run_dbt

    run_dbt("build", small_config, warehouse_db)
    return warehouse_db
