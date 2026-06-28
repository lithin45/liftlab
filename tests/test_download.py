"""Data acquisition + provenance/disclosure, the offline guarantee and its honesty.

These cover the exact paths a real run takes (auto -> synthetic fallback) and the
provenance edge cases the manifest must never get wrong (stale reuse, source switch,
unknown-origin CSVs), none of which the dbt fixtures exercise.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from liftlab.config import Config
from liftlab.data import download
from liftlab.data.download import ensure_raw_data
from liftlab.data.population import write_population_csvs
from liftlab.paths import OLIST_TABLES


def _with_source(cfg: Config, source: str) -> Config:
    return replace(cfg, data=replace(cfg.data, source=source))


@pytest.fixture
def no_kaggle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(download, "_kaggle_available", lambda: False)


def test_auto_falls_back_to_synthetic(small_config, tmp_path: Path, no_kaggle) -> None:
    """The default `auto` source must work offline by producing a disclosed synthetic pop."""
    cfg = _with_source(small_config, "auto")
    manifest = ensure_raw_data(cfg, raw_dir=tmp_path)

    assert manifest["source_used"] == "synthetic"
    assert manifest["is_synthetic_population"] is True
    assert "SYNTHETIC" in manifest["disclosure"]
    assert manifest["config_hash"] == cfg.config_hash()
    assert manifest["seed"] == cfg.seed
    assert manifest["files"] == [f"{n}.csv" for n in OLIST_TABLES]
    for name in OLIST_TABLES:
        assert (tmp_path / f"{name}.csv").is_file()
    assert (tmp_path / "MANIFEST.json").is_file()


def test_kaggle_required_raises_when_unavailable(small_config, tmp_path: Path, no_kaggle) -> None:
    cfg = _with_source(small_config, "kaggle")
    with pytest.raises(RuntimeError, match="kaggle"):
        ensure_raw_data(cfg, raw_dir=tmp_path)


def test_unknown_source_raises(small_config, tmp_path: Path, no_kaggle) -> None:
    cfg = _with_source(small_config, "bogus")
    with pytest.raises(ValueError, match="bogus"):
        ensure_raw_data(cfg, raw_dir=tmp_path)


def test_reuse_is_idempotent_for_matching_config(small_config, tmp_path: Path) -> None:
    cfg = _with_source(small_config, "synthetic")
    first = ensure_raw_data(cfg, raw_dir=tmp_path)
    second = ensure_raw_data(cfg, raw_dir=tmp_path)
    # Same config -> the stored manifest is returned verbatim (no regeneration).
    assert first == second


def test_changed_seed_invalidates_cache(small_config, tmp_path: Path) -> None:
    cfg = _with_source(small_config, "synthetic")
    first = ensure_raw_data(cfg, raw_dir=tmp_path)

    cfg2 = replace(cfg, seed=cfg.seed + 1)
    second = ensure_raw_data(cfg2, raw_dir=tmp_path)
    # config_hash changed -> rebuilt with the new seed, not silently reused.
    assert second["seed"] == cfg2.seed
    assert second["config_hash"] == cfg2.config_hash()
    assert second["config_hash"] != first["config_hash"]


def test_source_switch_does_not_reuse_stale_data(small_config, tmp_path: Path, no_kaggle) -> None:
    """Synthetic data must NOT be silently served when the user switches to kaggle."""
    ensure_raw_data(_with_source(small_config, "synthetic"), raw_dir=tmp_path)
    # Now request real Kaggle data (unavailable): must rebuild, not reuse synthetic.
    with pytest.raises(RuntimeError):
        ensure_raw_data(_with_source(small_config, "kaggle"), raw_dir=tmp_path)


def test_preexisting_csvs_without_manifest_are_disclosed_unknown(
    small_config, tmp_path: Path, no_kaggle
) -> None:
    """CSVs we did not generate (no manifest) must never be labelled REAL Olist."""
    write_population_csvs(small_config, tmp_path)  # 5 CSVs, but NO manifest
    cfg = _with_source(small_config, "auto")
    manifest = ensure_raw_data(cfg, raw_dir=tmp_path)

    assert manifest["source_used"] == "preexisting"
    assert manifest["is_synthetic_population"] is None  # unknown, not guessed
    assert "UNKNOWN" in manifest["disclosure"]
    # The CC BY-NC-SA / "Real Olist" attribution must NOT be asserted for unknown data.
    assert "CC BY-NC-SA" not in manifest["disclosure"]
