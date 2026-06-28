"""Phase 4: the SRM detector flags an intentionally imbalanced split and clears a balanced one."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as scs

from liftlab.config import load_config
from liftlab.simulation.simulate import simulate_experiment
from liftlab.srm.srm import srm_check


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def covariate() -> np.ndarray:
    rng = np.random.default_rng(20240601)
    return rng.lognormal(mean=3.5, sigma=0.8, size=20000)


def test_srm_chi_square_matches_scipy() -> None:
    n_c, n_t, ratio = 9000, 11000, 0.5
    res = srm_check(n_c, n_t, expected_ratio=ratio)
    total = n_c + n_t
    exp_t, exp_c = total * ratio, total * (1 - ratio)
    ref = scs.chisquare(f_obs=[n_t, n_c], f_exp=[exp_t, exp_c])
    assert res.chi_square == pytest.approx(ref.statistic, rel=1e-12)
    assert res.p_value == pytest.approx(ref.pvalue, rel=1e-9)


def test_srm_known_counts_analytic() -> None:
    # 10100 vs 9900 against 10000/10000 -> chi2 = 100^2/10000 * 2 = 2.0
    res = srm_check(n_control=9900, n_treatment=10100, expected_ratio=0.5)
    assert res.chi_square == pytest.approx(2.0, rel=1e-12)


def test_srm_flags_intentionally_imbalanced_assignment(cfg, covariate) -> None:
    """THE GATE: a 0.55 split (intended 0.50) at N=20k must be flagged."""
    res = simulate_experiment(
        cfg, covariate, seed=1, assignment_ratio=cfg.validation.srm_imbalance_ratio
    )
    check = srm_check(
        n_control=res.design["n_control"],
        n_treatment=res.design["n_treatment"],
        expected_ratio=cfg.experiment.assignment_ratio,
        threshold=cfg.validation.srm_alpha,
    )
    assert check.is_srm
    assert check.p_value < cfg.validation.srm_alpha


def test_srm_clears_a_balanced_assignment(cfg, covariate) -> None:
    res = simulate_experiment(cfg, covariate, seed=1)  # intended == realized == 0.5
    check = srm_check(
        n_control=res.design["n_control"],
        n_treatment=res.design["n_treatment"],
        expected_ratio=cfg.experiment.assignment_ratio,
        threshold=cfg.validation.srm_alpha,
    )
    assert not check.is_srm
    assert check.p_value > cfg.validation.srm_alpha


def test_srm_input_validation() -> None:
    with pytest.raises(ValueError):
        srm_check(100, 100, expected_ratio=0.0)
    with pytest.raises(ValueError):
        srm_check(100, 100, expected_ratio=1.0)
    with pytest.raises(ValueError):
        srm_check(0, 0, expected_ratio=0.5)
