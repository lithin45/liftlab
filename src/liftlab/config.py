"""Typed configuration loaded from ``config/experiment.yaml``.

The config is immutable and hashable: ``Config.config_hash()`` produces a stable
digest of the fully-resolved design, which we store with every run for
reproducibility/auditability. Selected fields can be overridden via environment
variables (used by CI to force the synthetic population).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from liftlab.paths import CONFIG_PATH


@dataclass(frozen=True)
class SyntheticConfig:
    n_customers: int
    start_date: str
    end_date: str


@dataclass(frozen=True)
class DataConfig:
    source: str  # "auto" | "kaggle" | "synthetic"
    kaggle_dataset: str
    cutoff_date: str
    synthetic: SyntheticConfig


@dataclass(frozen=True)
class ContinuousMetric:
    """A continuous outcome (e.g. revenue per user). ``true_effect_absolute`` is the
    injected ATE in outcome units."""

    name: str
    type: str  # "continuous"
    baseline_mean: float
    outcome_sd: float
    true_effect_absolute: float


@dataclass(frozen=True)
class ProportionMetric:
    """A binary outcome (e.g. conversion). ``true_lift_absolute`` is the injected ATE
    on the probability scale."""

    name: str
    type: str  # "proportion"
    base_rate: float
    true_lift_absolute: float
    covariate_coef: float


@dataclass(frozen=True)
class MetricsConfig:
    revenue: ContinuousMetric
    conversion: ProportionMetric


@dataclass(frozen=True)
class CovariateConfig:
    name: str
    target_correlation: float


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    unit: str
    assignment_ratio: float
    sample_size: int
    covariate: CovariateConfig
    metrics: MetricsConfig


@dataclass(frozen=True)
class PowerConfig:
    alpha: float
    power: float


@dataclass(frozen=True)
class ValidationConfig:
    n_simulations: int
    coverage_target: float
    coverage_floor: float
    coverage_mc_sigma: float
    aa_fpr_tolerance: float
    cuped_min_variance_reduction: float
    srm_imbalance_ratio: float
    srm_alpha: float


@dataclass(frozen=True)
class Config:
    seed: int
    data: DataConfig
    experiment: ExperimentConfig
    power: PowerConfig
    validation: ValidationConfig

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        """Stable 16-char digest of the fully-resolved configuration."""
        payload = json.dumps(self.as_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """Allow a few env vars to override the YAML (used by CI / Docker)."""
    if (source := os.environ.get("LIFTLAB_DATA_SOURCE")) is not None:
        raw.setdefault("data", {})["source"] = source
    if (seed := os.environ.get("LIFTLAB_SEED")) is not None:
        raw["seed"] = int(seed)
    return raw


def from_dict(raw: dict[str, Any]) -> Config:
    data = raw["data"]
    syn = data["synthetic"]
    exp = raw["experiment"]
    metrics = exp["metrics"]
    rev = metrics["revenue"]
    conv = metrics["conversion"]
    cov = exp["covariate"]
    val = raw["validation"]
    return Config(
        seed=int(raw["seed"]),
        data=DataConfig(
            source=str(data["source"]),
            kaggle_dataset=str(data["kaggle_dataset"]),
            cutoff_date=str(data["cutoff_date"]),
            synthetic=SyntheticConfig(
                n_customers=int(syn["n_customers"]),
                start_date=str(syn["start_date"]),
                end_date=str(syn["end_date"]),
            ),
        ),
        experiment=ExperimentConfig(
            name=str(exp["name"]),
            unit=str(exp["unit"]),
            assignment_ratio=float(exp["assignment_ratio"]),
            sample_size=int(exp["sample_size"]),
            covariate=CovariateConfig(
                name=str(cov["name"]),
                target_correlation=float(cov["target_correlation"]),
            ),
            metrics=MetricsConfig(
                revenue=ContinuousMetric(
                    name=str(rev["name"]),
                    type=str(rev["type"]),
                    baseline_mean=float(rev["baseline_mean"]),
                    outcome_sd=float(rev["outcome_sd"]),
                    true_effect_absolute=float(rev["true_effect_absolute"]),
                ),
                conversion=ProportionMetric(
                    name=str(conv["name"]),
                    type=str(conv["type"]),
                    base_rate=float(conv["base_rate"]),
                    true_lift_absolute=float(conv["true_lift_absolute"]),
                    covariate_coef=float(conv["covariate_coef"]),
                ),
            ),
        ),
        power=PowerConfig(
            alpha=float(raw["power"]["alpha"]),
            power=float(raw["power"]["power"]),
        ),
        validation=ValidationConfig(
            n_simulations=int(val["n_simulations"]),
            coverage_target=float(val["coverage_target"]),
            coverage_floor=float(val["coverage_floor"]),
            coverage_mc_sigma=float(val["coverage_mc_sigma"]),
            aa_fpr_tolerance=float(val["aa_fpr_tolerance"]),
            cuped_min_variance_reduction=float(val["cuped_min_variance_reduction"]),
            srm_imbalance_ratio=float(val["srm_imbalance_ratio"]),
            srm_alpha=float(val["srm_alpha"]),
        ),
    )


def load_config(path: Path | None = None) -> Config:
    """Load and validate the experiment configuration."""
    cfg_path = path or CONFIG_PATH
    with open(cfg_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw = _apply_env_overrides(raw)
    return from_dict(raw)
