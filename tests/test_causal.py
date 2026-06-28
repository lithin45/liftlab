"""Phase 6: the observational causal fallback recovers the known effect under confounding.

When randomization breaks (treatment depends on the pre-period confounder), the naive
post-period difference is biased. DiD and IPW must recover the INJECTED true effect; the
placebo (permuted treatment) must vanish.
"""

from __future__ import annotations

import numpy as np
import pytest

from liftlab.causal.estimators import (
    did_estimate,
    ipw_estimate,
    naive_estimate,
    placebo_effect,
    run_causal_analysis,
)
from liftlab.causal.scenario import make_confounded_scenario
from liftlab.config import load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def covariate() -> np.ndarray:
    rng = np.random.default_rng(20240601)
    return rng.lognormal(mean=3.5, sigma=0.8, size=6000)


@pytest.fixture(scope="module")
def scenario(cfg, covariate):
    return make_confounded_scenario(cfg, covariate, seed=1, confounding=1.0)


def test_scenario_structure(cfg, scenario) -> None:
    d = scenario.data
    assert list(d.columns) == ["unit_id", "confounder", "treatment", "y_pre", "y_post"]
    assert set(np.unique(d["treatment"])) <= {0, 1}
    assert 0 < scenario.n_treatment < len(d)
    assert scenario.true_effect == cfg.experiment.metrics.revenue.true_effect_absolute


def test_naive_estimate_is_biased(scenario) -> None:
    naive = naive_estimate(scenario.data)
    # Confounding pushes the naive estimate well away from the truth, and its CI misses it.
    assert abs(naive.estimate - scenario.true_effect) > 1.0
    assert not naive.ci_contains(scenario.true_effect)


def test_did_recovers_truth(scenario) -> None:
    did = did_estimate(scenario.data)
    naive = naive_estimate(scenario.data)
    assert did.ci_contains(scenario.true_effect)
    assert abs(did.estimate - scenario.true_effect) < abs(naive.estimate - scenario.true_effect)


def test_ipw_recovers_truth(scenario) -> None:
    ipw = ipw_estimate(scenario.data, n_boot=100, seed=0)
    naive = naive_estimate(scenario.data)
    assert ipw.ci_contains(scenario.true_effect)
    assert abs(ipw.estimate - scenario.true_effect) < abs(naive.estimate - scenario.true_effect)
    assert ipw.min_propensity > 0.0 and ipw.max_propensity < 1.0  # overlap maintained


def test_placebo_treatment_vanishes(scenario) -> None:
    naive_bias = abs(naive_estimate(scenario.data).estimate - scenario.true_effect)
    placebo = placebo_effect(scenario.data, seed=0)
    # A permuted treatment carries no real effect: far smaller than the confounding bias.
    assert abs(placebo) < 0.3 * naive_bias


def test_did_is_unbiased_over_seeds(cfg, covariate) -> None:
    truth = cfg.experiment.metrics.revenue.true_effect_absolute
    ests = [
        did_estimate(
            make_confounded_scenario(cfg, covariate, seed=s, confounding=1.0).data
        ).estimate
        for s in range(50)
    ]
    mean = float(np.mean(ests))
    se = float(np.std(ests, ddof=1) / np.sqrt(len(ests)))
    assert abs(mean - truth) < 4 * se


def test_ipw_is_unbiased_on_a_heavy_tailed_covariate(cfg) -> None:
    """Regression guard for the overlap fix: with an extremely right-skewed covariate
    (like the real pre-period spend), the confounder must be winsorized so the propensity
    stays in (0,1) and IPW recovers the truth, the unfixed bias was ~+0.25."""
    rng = np.random.default_rng(7)
    skewed = rng.lognormal(mean=3.0, sigma=1.2, size=6000)
    truth = cfg.experiment.metrics.revenue.true_effect_absolute
    ests = [
        ipw_estimate(
            make_confounded_scenario(cfg, skewed, seed=s, confounding=1.0).data, n_boot=80, seed=s
        ).estimate
        for s in range(40)
    ]
    assert abs(float(np.mean(ests)) - truth) < 0.20


@pytest.mark.slow
def test_dowhy_crosscheck_and_refuters(scenario) -> None:
    pytest.importorskip("dowhy")
    res = run_causal_analysis(
        scenario.data, scenario.true_effect, n_boot=50, seed=0, with_dowhy=True
    )
    assert res.dowhy_ipw is not None
    # DoWhy's propensity weighting agrees with the hand-rolled IPW (same method family).
    assert abs(res.dowhy_ipw - res.ipw.estimate) < 1.0
    assert res.refutations.get("random_common_cause") is not None
