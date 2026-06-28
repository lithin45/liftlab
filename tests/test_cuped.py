"""Phase 4: CUPED hits the >=30% variance-reduction gate, stays unbiased, tightens the CI."""

from __future__ import annotations

import numpy as np
import pytest

from liftlab.config import load_config
from liftlab.cuped.cuped import cuped_adjust, cuped_estimate, cuped_theta
from liftlab.simulation.simulate import simulate_experiment


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def covariate() -> np.ndarray:
    rng = np.random.default_rng(20240601)
    return rng.lognormal(mean=3.5, sigma=0.8, size=20000)


def test_cuped_theta_matches_cov_over_var() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    y = 3.0 * x + rng.normal(0, 1, 5000)
    expected = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    assert cuped_theta(y, x) == pytest.approx(expected, rel=1e-12)


def test_cuped_adjust_preserves_mean() -> None:
    """Centering X means the adjusted metric has the same overall mean (no bias)."""
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 5000)
    y = 2.0 + 3.0 * x + rng.normal(0, 1, 5000)
    y_cuped, _ = cuped_adjust(y, x)
    assert y_cuped.mean() == pytest.approx(y.mean(), abs=1e-9)


def test_cuped_meets_variance_reduction_gate(cfg, covariate) -> None:
    """THE GATE: CUPED variance reduction >= configured threshold (30%)."""
    res = simulate_experiment(cfg, covariate, seed=1)
    cuped = cuped_estimate(
        res.units["y_revenue"].to_numpy(),
        res.units["covariate"].to_numpy(),
        res.units["variant"].to_numpy(),
    )
    assert cuped.variance_reduction >= cfg.validation.cuped_min_variance_reduction
    # Pin the achieved reduction to the DESIGN's calibrated correlation (rho -> rho^2),
    # NOT to the self-referential VR == corr^2 algebraic identity (which always holds for
    # the optimal in-sample theta and would validate nothing).
    target_rho = cfg.experiment.covariate.target_correlation
    assert cuped.correlation == pytest.approx(target_rho, abs=0.02)
    assert cuped.variance_reduction == pytest.approx(target_rho**2, abs=0.02)


def test_cuped_preserves_effect_and_tightens_ci(cfg, covariate) -> None:
    res = simulate_experiment(cfg, covariate, seed=2)
    truth = res.design["revenue"]["effective_true_ate"]
    cuped = cuped_estimate(
        res.units["y_revenue"].to_numpy(),
        res.units["covariate"].to_numpy(),
        res.units["variant"].to_numpy(),
    )
    # Unbiased: adjusted estimate close to the naive one and both cover the truth.
    assert cuped.adjusted.ci_contains(truth)
    assert cuped.unadjusted.ci_contains(truth)
    # Tighter: CUPED shrinks the standard error and the CI width.
    assert cuped.adjusted.se < cuped.unadjusted.se
    assert cuped.adjusted.ci_width < cuped.unadjusted.ci_width
    assert cuped.se_reduction > 0.15  # ~ 1 - sqrt(1-rho^2) ~ 0.20


def test_cuped_is_unbiased_over_many_seeds(cfg, covariate) -> None:
    truth = cfg.experiment.metrics.revenue.true_effect_absolute
    k = 120
    ests = []
    for seed in range(200, 200 + k):
        res = simulate_experiment(cfg, covariate, seed=seed)
        cuped = cuped_estimate(
            res.units["y_revenue"].to_numpy(),
            res.units["covariate"].to_numpy(),
            res.units["variant"].to_numpy(),
        )
        ests.append(cuped.adjusted.estimate)
    mean_est = float(np.mean(ests))
    se_of_mean = float(np.std(ests, ddof=1) / np.sqrt(k))
    assert abs(mean_est - truth) < 4 * se_of_mean


@pytest.mark.slow
def test_cuped_estimate_has_correct_coverage(cfg) -> None:
    """The CUPED CI (Welch on Y_cuped, treating theta as known) must still cover the true
    effect ~95% of the time -- i.e. ignoring theta-estimation uncertainty does not break
    coverage at this N. This is the property the Phase-5 coverage gate relies on."""
    rng = np.random.default_rng(7)
    cov = rng.lognormal(mean=3.5, sigma=0.8, size=6000)
    truth = cfg.experiment.metrics.revenue.true_effect_absolute
    k = 2000
    cover_cuped = cover_naive = 0
    for seed in range(1000, 1000 + k):
        res = simulate_experiment(cfg, cov, seed=seed)
        c = cuped_estimate(
            res.units["y_revenue"].to_numpy(),
            res.units["covariate"].to_numpy(),
            res.units["variant"].to_numpy(),
        )
        cover_cuped += c.adjusted.ci_contains(truth)
        cover_naive += c.unadjusted.ci_contains(truth)
    assert abs(cover_cuped / k - 0.95) < 0.02
    assert abs(cover_naive / k - 0.95) < 0.02


def test_cuped_constant_covariate_is_a_noop(cfg, covariate) -> None:
    res = simulate_experiment(cfg, covariate, seed=3)
    const_cov = np.ones(res.units.shape[0])
    cuped = cuped_estimate(
        res.units["y_revenue"].to_numpy(), const_cov, res.units["variant"].to_numpy()
    )
    assert cuped.theta == 0.0
    assert cuped.variance_reduction == pytest.approx(0.0, abs=1e-12)
    assert cuped.adjusted.estimate == pytest.approx(cuped.unadjusted.estimate, rel=1e-12)
