"""Config loads, validates, and hashes stably."""

from __future__ import annotations

import pytest

from liftlab.config import Config, load_config


def test_config_loads_and_has_disclosed_truth() -> None:
    cfg = load_config()
    assert isinstance(cfg, Config)
    # Both injected ground-truth effects must be present and non-trivial.
    assert cfg.experiment.metrics.conversion.true_lift_absolute != 0.0
    assert cfg.experiment.metrics.revenue.true_effect_absolute != 0.0
    assert 0.0 < cfg.experiment.assignment_ratio < 1.0
    assert cfg.power.alpha == 0.05


def test_config_hash_is_stable() -> None:
    h1 = load_config().config_hash()
    h2 = load_config().config_hash()
    assert h1 == h2
    assert len(h1) == 16


def test_env_overrides_are_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    """CI sets LIFTLAB_DATA_SOURCE=synthetic to force the offline build; verify it works."""
    monkeypatch.setenv("LIFTLAB_DATA_SOURCE", "synthetic")
    monkeypatch.setenv("LIFTLAB_SEED", "12345")
    cfg = load_config()
    assert cfg.data.source == "synthetic"
    assert cfg.seed == 12345  # parsed as int


def test_no_env_leaves_yaml_intact(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LIFTLAB_DATA_SOURCE", raising=False)
    monkeypatch.delenv("LIFTLAB_SEED", raising=False)
    cfg = load_config()
    assert cfg.data.source == "auto"  # the YAML default
