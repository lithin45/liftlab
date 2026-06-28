"""Obtain the population substrate: real Olist via Kaggle, or synthetic fallback.

Resolution order is controlled by ``config.data.source``:

* ``"kaggle"``    : require the real Olist dataset; raise with manual instructions
                    if the Kaggle CLI / credentials are missing.
* ``"synthetic"`` : always generate the local synthetic population.
* ``"auto"``      : try Kaggle; transparently fall back to synthetic.

Every run writes ``data/raw/MANIFEST.json`` recording exactly which path was used,
whether the population is synthetic, the seed, and the config hash, so provenance
is never ambiguous and the synthetic design is always disclosed. Provenance is
never *guessed*: pre-existing CSVs with no manifest are disclosed as UNKNOWN, never
as real Olist.

Reuse is config-aware: cached CSVs are reused only when the stored manifest's
``config_hash`` and ``source_used`` match the currently-requested config. This means
switching ``data.source`` synthetic -> kaggle (or changing the seed) correctly
rebuilds instead of silently serving stale data.

DATA LICENSING: the real Olist dataset is CC BY-NC-SA 4.0 (NON-COMMERCIAL). LiftLab
uses it only as a non-commercial population substrate and attributes it (DATA_LICENSE.md).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from liftlab.config import Config
from liftlab.data import population
from liftlab.paths import OLIST_TABLES, RAW_DIR

MANIFEST_NAME = "MANIFEST.json"

_KAGGLE_MANUAL_HINT = (
    "Real Olist data requires the Kaggle CLI and an API token.\n"
    "  1. pip install kaggle  (or: uv add --group dev kaggle)\n"
    "  2. Create ~/.kaggle/kaggle.json from https://www.kaggle.com/settings (API > Create New Token)\n"
    "  3. chmod 600 ~/.kaggle/kaggle.json\n"
    "  4. Re-run `make data-force`.\n"
    "Or set data.source: synthetic in config/experiment.yaml to use the offline "
    "synthetic population (clearly disclosed)."
)

# Disclosure string keyed by the resolved source. 'preexisting' = provenance unknown.
_DISCLOSURE = {
    "synthetic": "SYNTHETIC population substrate, not real Olist data.",
    "kaggle": "Real Olist dataset (Kaggle, CC BY-NC-SA 4.0, non-commercial). See DATA_LICENSE.md.",
    "preexisting": (
        "Provenance UNKNOWN, pre-existing CSVs with no recorded manifest; could be real "
        "Olist or a synthetic stand-in. Re-run `make data-force` for a trustworthy manifest."
    ),
}


def _expected_files_present(raw_dir: Path) -> bool:
    return all((raw_dir / f"{name}.csv").is_file() for name in OLIST_TABLES)


def read_manifest(raw_dir: Path | None = None) -> dict | None:
    """Return the stored provenance manifest, or None if absent."""
    raw_dir = raw_dir or RAW_DIR
    path = raw_dir / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def population_is_synthetic(raw_dir: Path | None = None) -> bool | None:
    """True/False if provenance is recorded, else None (unknown)."""
    manifest = read_manifest(raw_dir)
    return None if manifest is None else manifest.get("is_synthetic_population")


def _kaggle_available() -> bool:
    if shutil.which("kaggle") is None:
        return False
    has_token = (Path.home() / ".kaggle" / "kaggle.json").is_file()
    has_env = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    return has_token or has_env


def _try_kaggle(config: Config, raw_dir: Path) -> bool:
    """Attempt the Kaggle download. Returns True on success."""
    if not _kaggle_available():
        return False
    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                config.data.kaggle_dataset,
                "-p",
                str(raw_dir),
                "--unzip",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=900,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"[liftlab] Kaggle download failed: {exc}")
        return False
    return _expected_files_present(raw_dir)


def _write_manifest(raw_dir: Path, manifest: dict) -> None:
    with open(raw_dir / MANIFEST_NAME, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)


def _record_manifest(
    raw_dir: Path,
    config: Config,
    source_used: str,
    is_synthetic: bool | None,
    counts: dict[str, int] | None,
) -> dict:
    """Build, persist, and return the provenance manifest."""
    manifest = {
        "source_used": source_used,
        "is_synthetic_population": is_synthetic,  # True | False | None(unknown)
        "seed": config.seed,
        "config_hash": config.config_hash(),
        "row_counts": counts,
        "files": [f"{name}.csv" for name in OLIST_TABLES],
        "generated_at": datetime.now(UTC).isoformat(),
        "disclosure": _DISCLOSURE[source_used],
        "treatment_note": (
            "The experiment treatment effect is ALWAYS synthetic/injected; the true "
            "effect is known and disclosed in the report card and README."
        ),
    }
    _write_manifest(raw_dir, manifest)
    return manifest


def _manifest_matches_request(manifest: dict, config: Config) -> bool:
    """Whether cached CSVs may be reused: same config_hash AND a source consistent
    with the currently-requested ``data.source``."""
    if manifest.get("config_hash") != config.config_hash():
        return False
    used = manifest.get("source_used")
    source = config.data.source.lower()
    if source == "synthetic":
        return used == "synthetic"
    if source == "kaggle":
        return used == "kaggle"
    if source == "auto":
        return used in {"kaggle", "synthetic"}
    return False


def _build_for_source(
    config: Config, raw_dir: Path
) -> tuple[str, bool | None, dict[str, int] | None]:
    """(Re)produce the raw CSVs for the requested source. Returns (source_used, is_synthetic, counts)."""
    source = config.data.source.lower()
    if source == "synthetic":
        return "synthetic", True, population.write_population_csvs(config, raw_dir)
    if source in {"kaggle", "auto"}:
        if _try_kaggle(config, raw_dir):
            return "kaggle", False, None
        if source == "kaggle":
            raise RuntimeError(
                "data.source=kaggle but the Kaggle dataset could not be obtained.\n"
                + _KAGGLE_MANUAL_HINT
            )
        print(
            "[liftlab] Kaggle unavailable; falling back to the SYNTHETIC population "
            "(offline, disclosed)."
        )
        return "synthetic", True, population.write_population_csvs(config, raw_dir)
    raise ValueError(f"Unknown data.source: {config.data.source!r}")


def ensure_raw_data(config: Config, raw_dir: Path | None = None, force: bool = False) -> dict:
    """Ensure the five Olist-shaped CSVs exist in ``raw_dir``. Returns the manifest.

    Reuse rules (when ``force`` is False):
      * CSVs + a manifest whose config_hash AND source match the request -> reuse as-is.
      * CSVs + a stale/mismatched manifest -> rebuild for the requested source.
      * CSVs + NO manifest -> provenance unknown; for ``synthetic`` we regenerate
        (cheap, deterministic), otherwise we keep the files but disclose them as
        UNKNOWN rather than guessing they are real Olist.
    """
    raw_dir = raw_dir or RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = raw_dir / MANIFEST_NAME
    files_present = _expected_files_present(raw_dir)

    if files_present and not force:
        if manifest_path.is_file():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing = None
            if existing and _manifest_matches_request(existing, config):
                return existing
            # Stale / source-mismatch -> fall through and rebuild for the request.
        elif config.data.source.lower() != "synthetic":
            # CSVs we did not generate and cannot verify: disclose as UNKNOWN, never "real".
            return _record_manifest(raw_dir, config, "preexisting", None, None)
        # else: synthetic source + no manifest -> regenerate below for a clean manifest.

    source_used, is_synthetic, counts = _build_for_source(config, raw_dir)
    return _record_manifest(raw_dir, config, source_used, is_synthetic, counts)
