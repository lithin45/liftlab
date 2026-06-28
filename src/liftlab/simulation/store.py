"""Persist and reload a simulated experiment (the per-unit table + its design).

Each run lands in ``runs/<run_id>/`` with:
  * ``design.json``, the disclosed ground truth. A *deterministic* function of
                          (config, seed, call-time overrides), no wall-clock fields,
                          so re-running the same design is byte-idempotent.
  * ``units.csv``, the per-unit experiment table (round-trip exact).
  * ``provenance.json``, non-deterministic audit metadata (wall-clock timestamp).

The run_id encodes the *realized* design variant (assignment ratio + injected effects),
not just config hash + seed, so SRM / A/A / partial-effect variants never collide.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from liftlab.paths import RUNS_DIR
from liftlab.simulation.simulate import SimulationResult

DESIGN_FILE = "design.json"
UNITS_FILE = "units.csv"
PROVENANCE_FILE = "provenance.json"


def run_id(design: dict) -> str:
    """Deterministic, collision-free run identifier.

    Includes the realized assignment ratio and injected effects so that an SRM run
    (different ratio) or a partial/full A/A run (different effects) at the same seed
    gets a distinct directory instead of silently overwriting another run.
    """
    variant_key = "|".join(
        [
            f"{design['assignment_ratio_intended']:.4f}",
            f"{design['revenue']['true_effect_absolute']:.6g}",
            f"{design['conversion']['true_lift_absolute']:.6g}",
        ]
    )
    tag = hashlib.sha1(variant_key.encode()).hexdigest()[:8]
    suffix = "_aa" if design.get("is_aa") else ""
    return f"{design['config_hash']}_seed{design['seed']}_{tag}{suffix}"


def store_run(result: SimulationResult, runs_dir: Path | None = None) -> Path:
    """Write design.json + units.csv + provenance.json for a run. Returns the run dir."""
    runs_dir = runs_dir or RUNS_DIR
    out = runs_dir / run_id(result.design)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / DESIGN_FILE, "w", encoding="utf-8") as fh:
        json.dump(result.design, fh, indent=2, sort_keys=True)
    # to_csv emits full float repr; the read side pins round-trip precision (see load_run).
    result.units.to_csv(out / UNITS_FILE, index=False)
    with open(out / PROVENANCE_FILE, "w", encoding="utf-8") as fh:
        json.dump(
            {"run_id": out.name, "generated_at": datetime.now(UTC).isoformat()},
            fh,
            indent=2,
        )
    return out


def load_run(run_dir: Path) -> SimulationResult:
    """Reload a stored run. The units table reloads bit-exact (round-trip float parse)."""
    run_dir = Path(run_dir)
    with open(run_dir / DESIGN_FILE, encoding="utf-8") as fh:
        design = json.load(fh)
    units = pd.read_csv(run_dir / UNITS_FILE, float_precision="round_trip")
    return SimulationResult(units=units, design=design)
