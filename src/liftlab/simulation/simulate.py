"""Synthetic randomized-experiment generator (semi-synthetic, calibrated).

DISCLOSURE: the treatment effect here is INJECTED by us, so the true effect is known.
This is the entire basis for LiftLab's validation: every estimator is checked against
this known truth.

Design (chosen with the user in Phase 2)
----------------------------------------
* **Units** are drawn from the real (or synthetic-stand-in) population; each unit's
  pre-period spend (``pre_period_value`` from the ``customer_metrics`` mart) is the
  **CUPED covariate** X. We standardize it to z (mean 0, var 1).
* **Assignment** T ~ Bernoulli(ratio), seeded.
* **Continuous outcome** (revenue per user), calibrated so corr(z, Y) == target rho:

      Y = mu + beta*z + tau*T + eps,   beta = rho*s,   eps ~ N(0, (1 - rho^2)*s^2)

  With Var(z)=1 this gives corr(z, Y) ~= rho (exact when tau=0), so CUPED's variance
  reduction == rho^2 (~36% for rho=0.6) **by construction** -- and the true ATE is tau.
* **Binary outcome** (conversion):  p = base_rate + tau_p*T + delta*z  (clipped to (0,1)),
  Y ~ Bernoulli(p).  Since E[z]=0, the true ATE is exactly tau_p (clipping is negligible).

Everything derives from a single seed -> identical draws are reproducible. Monte-Carlo
replicates (Phase 5) re-randomize assignment + redraw outcome noise over the same units
by passing ``seed = base_seed + i``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from liftlab.config import Config
from liftlab.paths import duckdb_path

# Columns of the per-unit experiment table.
UNIT_COLUMNS = ("unit_id", "variant", "covariate", "covariate_raw", "y_revenue", "y_conversion")


@dataclass
class SimulationResult:
    """A single simulated experiment: the per-unit table + the (disclosed) design."""

    units: pd.DataFrame
    design: dict


def load_covariate_values(db_path: Path | None = None) -> np.ndarray:
    """Read the pre-period covariate (pre_period_value) from the customer_metrics mart.

    Ordered by customer_unique_id so the returned array is deterministic (DuckDB does not
    otherwise guarantee row order), required for reproducible experiment draws.
    """
    import duckdb

    db_path = db_path or duckdb_path()
    if not Path(db_path).is_file():
        raise FileNotFoundError(
            f"Warehouse not found at {db_path}. Run `make data` first to build customer_metrics."
        )
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        values = con.execute(
            "SELECT pre_period_value FROM main.customer_metrics ORDER BY customer_unique_id"
        ).fetchnumpy()["pre_period_value"]
    finally:
        con.close()
    return np.asarray(values, dtype=float)


def sample_covariate(values: np.ndarray, sample_size: int, seed: int) -> np.ndarray:
    """Draw ``sample_size`` unit covariates from the population covariate distribution.

    Without replacement when the population is large enough, else with replacement.
    """
    rng = np.random.default_rng(seed)
    n_pop = len(values)
    replace = sample_size > n_pop
    idx = rng.choice(n_pop, size=sample_size, replace=replace)
    return values[idx]


def sample_population_units(
    config: Config, db_path: Path | None = None, *, seed: int | None = None
) -> np.ndarray:
    """Draw the experiment's unit covariates from the warehouse population.

    For Monte-Carlo, keep ``seed`` fixed (default ``config.seed``) so the same units
    are reused across replicates while only assignment/outcome vary.
    """
    seed = config.seed if seed is None else seed
    population = load_covariate_values(db_path)
    return sample_covariate(population, config.experiment.sample_size, seed)


def _standardize(x: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance (population std). Returns zeros for a constant input."""
    mean = float(np.mean(x))
    std = float(np.std(x))  # ddof=0 -> Var(z) == 1 exactly
    if std < 1e-12:
        return np.zeros_like(x)
    return (x - mean) / std


def standardize_covariate(x: np.ndarray) -> np.ndarray:
    """Public standardization (mean 0, var 1) shared by the simulator and MC harness."""
    return _standardize(np.asarray(x, dtype=float))


@dataclass
class ExperimentDraw:
    """One replicate's per-unit arrays (no DataFrame) + its variant parameters."""

    variant: np.ndarray
    y_revenue: np.ndarray
    y_conversion: np.ndarray
    ratio: float
    revenue_effect: float
    conversion_lift: float


def _revenue_coeffs(config: Config) -> tuple[float, float]:
    """(beta, noise_sd) calibrating corr(z, revenue) == target_correlation."""
    rho = config.experiment.covariate.target_correlation
    s = config.experiment.metrics.revenue.outcome_sd
    return rho * s, math.sqrt(max(0.0, 1.0 - rho**2)) * s


def draw_experiment(
    config: Config,
    z: np.ndarray,
    *,
    seed: int,
    assignment_ratio: float | None = None,
    revenue_effect: float | None = None,
    conversion_lift: float | None = None,
) -> ExperimentDraw:
    """Generate one replicate's outcome arrays from a STANDARDIZED covariate ``z``.

    The fast path used by the Monte-Carlo harness (no DataFrame). The RNG consumption
    order matches :func:`simulate_experiment` exactly, so a seed yields identical draws.
    """
    rng = np.random.default_rng(seed)
    n = z.size
    ratio = config.experiment.assignment_ratio if assignment_ratio is None else assignment_ratio
    variant = (rng.random(n) < ratio).astype(int)

    rev = config.experiment.metrics.revenue
    beta, noise_sd = _revenue_coeffs(config)
    tau = rev.true_effect_absolute if revenue_effect is None else revenue_effect
    y_revenue = rev.baseline_mean + beta * z + tau * variant + rng.normal(0.0, noise_sd, n)

    conv = config.experiment.metrics.conversion
    tau_p = conv.true_lift_absolute if conversion_lift is None else conversion_lift
    p = np.clip(conv.base_rate + tau_p * variant + conv.covariate_coef * z, 1e-6, 1.0 - 1e-6)
    y_conversion = (rng.random(n) < p).astype(int)

    return ExperimentDraw(variant, y_revenue, y_conversion, float(ratio), float(tau), float(tau_p))


def conversion_effective_ate(
    config: Config, z: np.ndarray, conversion_lift: float | None = None
) -> float:
    """Conditional true ATE for conversion on covariate ``z`` (robust to clipping)."""
    conv = config.experiment.metrics.conversion
    tau_p = conv.true_lift_absolute if conversion_lift is None else conversion_lift
    lo, hi = 1e-6, 1.0 - 1e-6
    p_treat = np.clip(conv.base_rate + tau_p + conv.covariate_coef * z, lo, hi)
    p_ctrl = np.clip(conv.base_rate + conv.covariate_coef * z, lo, hi)
    return float(p_treat.mean() - p_ctrl.mean())


def simulate_experiment(
    config: Config,
    covariate_values: np.ndarray,
    *,
    seed: int,
    assignment_ratio: float | None = None,
    revenue_effect: float | None = None,
    conversion_lift: float | None = None,
) -> SimulationResult:
    """Generate one experiment from the supplied unit covariates.

    Parameters
    ----------
    covariate_values:
        Raw pre-period covariate per unit (length N). Use :func:`sample_covariate`
        to draw these from the population.
    seed:
        Replicate seed (assignment + outcome noise derive from it).
    assignment_ratio:
        Override the intended P(treatment), used to inject an SRM in Phase 4.
    revenue_effect, conversion_lift:
        Override the injected true effects, set both to 0.0 for an A/A simulation.
    """
    x = np.asarray(covariate_values, dtype=float)
    z = standardize_covariate(x)
    n = x.size

    draw = draw_experiment(
        config,
        z,
        seed=seed,
        assignment_ratio=assignment_ratio,
        revenue_effect=revenue_effect,
        conversion_lift=conversion_lift,
    )
    variant, y_revenue, y_conversion = draw.variant, draw.y_revenue, draw.y_conversion
    ratio, tau, tau_p = draw.ratio, draw.revenue_effect, draw.conversion_lift

    rev = config.experiment.metrics.revenue
    conv = config.experiment.metrics.conversion
    rho = config.experiment.covariate.target_correlation
    s = rev.outcome_sd
    beta, noise_sd = _revenue_coeffs(config)
    delta = conv.covariate_coef
    conv_ate = conversion_effective_ate(config, z, conversion_lift=conversion_lift)

    units = pd.DataFrame(
        {
            "unit_id": np.arange(n, dtype=np.int64),
            "variant": variant,
            "covariate": z,
            "covariate_raw": x,
            "y_revenue": y_revenue,
            "y_conversion": y_conversion,
        }
    )

    n_treatment = int(variant.sum())
    is_aa = tau == 0.0 and tau_p == 0.0
    design = {
        "name": config.experiment.name,
        "unit": config.experiment.unit,
        "seed": int(seed),
        "config_hash": config.config_hash(),
        "sample_size": int(n),
        "is_aa": is_aa,
        "assignment_ratio_intended": float(ratio),
        "n_treatment": n_treatment,
        "n_control": int(n - n_treatment),
        "realized_ratio": float(n_treatment / n) if n else 0.0,
        "target_correlation": float(rho),
        # Disclosed ground truth + the exact generative parameters. ``effective_true_ate``
        # is the conditional ATE on the realized units (== nominal for revenue; ~nominal
        # for conversion, exact under no clipping) and is what coverage is checked against.
        "revenue": {
            "name": rev.name,
            "true_effect_absolute": float(tau),
            "effective_true_ate": float(tau),  # linear, no clipping -> exact
            "baseline_mean": float(rev.baseline_mean),
            "outcome_sd": float(s),
            "beta": float(beta),
            "noise_sd": float(noise_sd),
        },
        "conversion": {
            "name": conv.name,
            "true_lift_absolute": float(tau_p),
            "effective_true_ate": conv_ate,
            "base_rate": float(conv.base_rate),
            "covariate_coef": float(delta),
        },
    }
    return SimulationResult(units=units, design=design)
