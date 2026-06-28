"""Load the raw Olist CSVs into DuckDB under a ``raw`` schema.

This is the "EL" of ELT: raw files land as VARCHAR-typed tables (robust against
messy real-world fields like review comments with embedded commas/newlines), and
dbt does the typed "T" in staging. Keeping raw as strings means the dbt staging
casts are explicit and visible, the SQL skill on display.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from liftlab.paths import OLIST_TABLES, RAW_DIR, duckdb_path

RAW_SCHEMA = "raw"


def load_raw_to_duckdb(db_path: Path | None = None, raw_dir: Path | None = None) -> dict[str, int]:
    """(Re)build the ``raw`` schema from the CSVs. Returns row counts per table."""
    db_path = db_path or duckdb_path()
    raw_dir = raw_dir or RAW_DIR
    db_path.parent.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    con = duckdb.connect(str(db_path))
    try:
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {RAW_SCHEMA}")
        for name in OLIST_TABLES:
            csv_path = raw_dir / f"{name}.csv"
            if not csv_path.is_file():
                raise FileNotFoundError(f"Expected raw CSV not found: {csv_path}")
            con.execute(
                f"CREATE OR REPLACE TABLE {RAW_SCHEMA}.{name} AS "
                "SELECT * FROM read_csv_auto(?, header=true, all_varchar=true)",
                [str(csv_path)],
            )
            counts[name] = con.execute(f"SELECT count(*) FROM {RAW_SCHEMA}.{name}").fetchone()[0]
    finally:
        con.close()
    return counts
