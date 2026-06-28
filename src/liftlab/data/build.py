"""Orchestrate the data layer: population -> DuckDB raw -> dbt fact tables.

``make data`` -> ``liftlab build`` runs this end to end:
  1. ensure the raw CSVs exist (Kaggle or synthetic),
  2. load them into DuckDB's ``raw`` schema,
  3. run ``dbt build`` (staging views + the ``customer_metrics`` mart + dbt tests),
  4. sanity-check the mart.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import duckdb

from liftlab.config import Config, load_config
from liftlab.data.download import ensure_raw_data
from liftlab.data.warehouse import load_raw_to_duckdb
from liftlab.paths import DBT_DIR, duckdb_path


def _dbt_executable() -> str:
    """Resolve the dbt entry point from the active venv (works under uv/Docker)."""
    exe = shutil.which("dbt")
    if exe:
        return exe
    candidate = Path(sys.executable).parent / "dbt"
    if candidate.is_file():
        return str(candidate)
    raise RuntimeError("dbt executable not found; is dbt-duckdb installed?")


def run_dbt(
    command: str,
    config: Config,
    db_path: Path | None = None,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess:
    """Run a dbt command against the project with the warehouse + vars wired in."""
    db_path = db_path or duckdb_path()
    env = os.environ.copy()
    env["LIFTLAB_DUCKDB"] = str(db_path)
    env["DBT_PROFILES_DIR"] = str(DBT_DIR)

    args = [
        _dbt_executable(),
        command,
        "--project-dir",
        str(DBT_DIR),
        "--profiles-dir",
        str(DBT_DIR),
        "--vars",
        f'{{"cutoff_date": "{config.data.cutoff_date}"}}',
    ]
    if extra_args:
        args.extend(extra_args)

    return subprocess.run(args, env=env, check=True, text=True)


def build(
    config: Config | None = None,
    force_data: bool = False,
    raw_dir: Path | None = None,
    db_path: Path | None = None,
) -> dict:
    """Run the full data build. Returns a summary dict.

    ``raw_dir`` / ``db_path`` default to the project locations; they are parameters
    so tests can run the whole orchestration hermetically in a temp directory.
    """
    config = config or load_config()
    db_path = db_path or duckdb_path()

    print("[liftlab] Step 1/3: ensuring population data ...")
    manifest = ensure_raw_data(config, raw_dir=raw_dir, force=force_data)
    disclosure = {True: "SYNTHETIC population", False: "REAL Olist population"}.get(
        manifest["is_synthetic_population"], "UNKNOWN-provenance population"
    )
    print(f"[liftlab]   -> source={manifest['source_used']} ({disclosure})")

    print("[liftlab] Step 2/3: loading raw CSVs into DuckDB ...")
    raw_counts = load_raw_to_duckdb(db_path, raw_dir)
    for name, count in raw_counts.items():
        print(f"[liftlab]   -> raw.{name}: {count:,} rows")

    print("[liftlab] Step 3/3: running dbt build (staging + marts + tests) ...")
    run_dbt("build", config, db_path)

    # Sanity-check the smoke-test fact table.
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        n_customers = con.execute("SELECT count(*) FROM main.customer_metrics").fetchone()[0]
        total_value = con.execute(
            "SELECT round(sum(total_value), 2) FROM main.customer_metrics"
        ).fetchone()[0]
    finally:
        con.close()

    print(
        f"[liftlab] Done. customer_metrics: {n_customers:,} customers, "
        f"total_value={total_value:,.2f}"
    )
    return {
        "manifest": manifest,
        "raw_counts": raw_counts,
        "customer_metrics_rows": n_customers,
        "customer_metrics_total_value": total_value,
    }
