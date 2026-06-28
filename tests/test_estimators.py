"""Phase 3: estimators match analytic / statsmodels truth on known cases.

The two-sample tests are validated against scipy/statsmodels (which are themselves
the analytic reference), the continuous power against statsmodels power classes, the
proportion power against a Monte-Carlo rejection-rate simulation, and every MDE/N
calculation against its own power formula (the inversion must round-trip).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as scs
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportions_ztest

from liftlab.estimators.power import (
    mde_two_means,
    mde_two_proportions,
    power_two_means,
    power_two_proportions,
    required_total_n_two_means,
    required_total_n_two_proportions,
)
from liftlab.estimators.twosample import (
    estimate_effect,
    two_proportion_ztest,
    welch_ttest,
)

# --------------------------------------------------------------------------- #
# Two-sample tests vs analytic truth
# --------------------------------------------------------------------------- #


def test_welch_matches_scipy() -> None:
    rng = np.random.default_rng(1)
    t = rng.normal(5.0, 2.0, 200)
    c = rng.normal(4.0, 3.0, 150)
    est = welch_ttest(t, c)
    ref = scs.ttest_ind(t, c, equal_var=False)
    ci = ref.confidence_interval(0.95)

    assert est.estimate == pytest.approx(t.mean() - c.mean(), rel=1e-12)
    assert est.statistic == pytest.approx(ref.statistic, rel=1e-9)
    assert est.p_value == pytest.approx(ref.pvalue, rel=1e-9)
    assert est.dof == pytest.approx(ref.df, rel=1e-9)
    assert est.ci_low == pytest.approx(ci.low, rel=1e-9)
    assert est.ci_high == pytest.approx(ci.high, rel=1e-9)


def test_welch_ci_is_estimate_plus_minus_tcrit_se() -> None:
    rng = np.random.default_rng(2)
    est = welch_ttest(rng.normal(0, 1, 50), rng.normal(0, 1, 60), alpha=0.10)
    crit = scs.t.ppf(0.95, est.dof)
    assert est.ci_low == pytest.approx(est.estimate - crit * est.se, rel=1e-12)
    assert est.ci_high == pytest.approx(est.estimate + crit * est.se, rel=1e-12)


def test_two_proportion_matches_statsmodels() -> None:
    n_t, s_t, n_c, s_c = 1200, 300, 1000, 220
    est = two_proportion_ztest(n_t, s_t, n_c, s_c)
    z_sm, p_sm = proportions_ztest([s_t, s_c], [n_t, n_c])  # pooled, two-sided
    assert est.statistic == pytest.approx(z_sm, rel=1e-9)
    assert est.p_value == pytest.approx(p_sm, rel=1e-9)
    assert est.estimate == pytest.approx(s_t / n_t - s_c / n_c, rel=1e-12)


def test_two_proportion_ci_is_agresti_caffo() -> None:
    n_t, s_t, n_c, s_c = 500, 300, 500, 250
    est = two_proportion_ztest(n_t, s_t, n_c, s_c)
    pt_adj = (s_t + 1) / (n_t + 2)
    pc_adj = (s_c + 1) / (n_c + 2)
    se = np.sqrt(pt_adj * (1 - pt_adj) / (n_t + 2) + pc_adj * (1 - pc_adj) / (n_c + 2))
    crit = scs.norm.ppf(0.975)
    center = pt_adj - pc_adj
    assert est.ci_low == pytest.approx(center - crit * se, rel=1e-12)
    assert est.ci_high == pytest.approx(center + crit * se, rel=1e-12)
    # Point estimate is still the raw difference.
    assert est.estimate == pytest.approx(s_t / n_t - s_c / n_c, rel=1e-12)


def test_two_proportion_ci_never_degenerate() -> None:
    """Wald would give a zero-width CI for these; Agresti-Caffo must not."""
    for n_t, s_t, n_c, s_c in [(100, 100, 100, 100), (100, 0, 100, 0), (50, 50, 50, 0)]:
        est = two_proportion_ztest(n_t, s_t, n_c, s_c)
        assert est.ci_high > est.ci_low


def test_estimate_effect_dispatch() -> None:
    rng = np.random.default_rng(3)
    n = 2000
    variant = rng.integers(0, 2, n)
    cont = rng.normal(0, 1, n)
    binar = rng.integers(0, 2, n).astype(float)
    assert estimate_effect(cont, variant, "continuous").method == "welch_ttest"
    assert estimate_effect(binar, variant, "proportion").method == "two_proportion_ztest"
    with pytest.raises(ValueError, match="metric_type"):
        estimate_effect(cont, variant, "nonsense")


# --------------------------------------------------------------------------- #
# Power / MDE (continuous) vs statsmodels
# --------------------------------------------------------------------------- #


def test_power_two_means_matches_normalindpower() -> None:
    effect, sd, n_t, n_c, alpha = 0.3, 1.0, 800, 600, 0.05
    mine = power_two_means(effect, sd, n_t, n_c, alpha)
    sm = NormalIndPower().power(
        effect_size=effect / sd, nobs1=n_t, alpha=alpha, ratio=n_c / n_t, alternative="two-sided"
    )
    assert mine == pytest.approx(sm, abs=1e-9)


def test_power_two_means_close_to_ttestpower() -> None:
    # Mid-range power + large N so the normal approx and the t-based power nearly agree.
    effect, sd, n_t, n_c, alpha = 0.04, 1.0, 5000, 5000, 0.05
    mine = power_two_means(effect, sd, n_t, n_c, alpha)
    sm_t = TTestIndPower().power(
        effect_size=effect / sd, nobs1=n_t, alpha=alpha, ratio=n_c / n_t, alternative="two-sided"
    )
    assert 0.4 < mine < 0.7  # genuinely mid-range, not saturated
    assert mine == pytest.approx(sm_t, abs=2e-3)  # normal vs t, negligible at large N


def test_mde_two_means_inverts_to_target_power() -> None:
    sd, n_t, n_c, alpha, power = 40.0, 10000, 10000, 0.05, 0.80
    mde = mde_two_means(sd, n_t, n_c, alpha, power)
    assert power_two_means(mde, sd, n_t, n_c, alpha) == pytest.approx(power, abs=1e-4)


def test_required_n_two_means_matches_statsmodels_and_inverts() -> None:
    sd, mde, alpha, power = 40.0, 2.0, 0.05, 0.80
    total = required_total_n_two_means(mde, sd, alpha, power, ratio=0.5)
    # statsmodels solves per-arm nobs1 for the same standardized effect.
    nobs1 = NormalIndPower().solve_power(
        effect_size=mde / sd, alpha=alpha, power=power, ratio=1.0, alternative="two-sided"
    )
    # ~2e-6 relative gap: the closed form drops the negligible opposite-tail term that
    # statsmodels keeps when numerically inverting the two-sided power.
    assert total == pytest.approx(2 * nobs1, rel=1e-5)
    # And the design hits the target power.
    assert power_two_means(mde, sd, total / 2, total / 2, alpha) == pytest.approx(power, abs=1e-4)


# --------------------------------------------------------------------------- #
# Power / MDE (proportions) vs Monte-Carlo + inversion
# --------------------------------------------------------------------------- #


def test_power_two_proportions_matches_simulation() -> None:
    # Mid-range power so the MC comparison can actually discriminate a wrong formula
    # (a saturated ~1.0 power is insensitive to pooled/unpooled or one/two-sided errors).
    p_c, p_t, n_t, n_c, alpha = 0.20, 0.235, 1500, 1500, 0.05
    analytic = power_two_proportions(p_c, p_t, n_t, n_c, alpha)
    assert 0.5 < analytic < 0.75  # genuinely mid-range
    rng = np.random.default_rng(0)
    k = 5000
    rejects = 0
    for _ in range(k):
        s_t = rng.binomial(n_t, p_t)
        s_c = rng.binomial(n_c, p_c)
        rejects += two_proportion_ztest(n_t, s_t, n_c, s_c, alpha).significant
    empirical = rejects / k
    assert abs(empirical - analytic) < 0.025  # ~3.5 MC-SE


def test_mde_two_proportions_inverts() -> None:
    p_c, n_t, n_c, alpha, power = 0.20, 5000, 5000, 0.05, 0.80
    mde = mde_two_proportions(p_c, n_t, n_c, alpha, power)
    assert power_two_proportions(p_c, p_c + mde, n_t, n_c, alpha) == pytest.approx(power, abs=1e-6)


def test_required_n_two_proportions_inverts() -> None:
    p_c, mde, alpha, power = 0.20, 0.02, 0.05, 0.80
    total = required_total_n_two_proportions(mde, p_c, alpha, power, ratio=0.5)
    assert power_two_proportions(p_c, p_c + mde, total / 2, total / 2, alpha) == pytest.approx(
        power, abs=1e-6
    )


def test_power_increases_with_n() -> None:
    """Sanity monotonicity: more data -> more power."""
    small = power_two_means(0.1, 1.0, 500, 500)
    big = power_two_means(0.1, 1.0, 5000, 5000)
    assert big > small


# --------------------------------------------------------------------------- #
# Edge cases + input validation (no silent garbage / crashes)
# --------------------------------------------------------------------------- #


def test_welch_zero_variance_separated_groups() -> None:
    est = welch_ttest([6, 6, 6, 6], [5, 5, 5])
    assert est.estimate == 1.0
    assert math.isinf(est.statistic)
    assert est.p_value == 0.0  # perfectly separated -> infinitely significant
    assert est.ci_low == est.ci_high == 1.0


def test_welch_zero_variance_identical_groups() -> None:
    est = welch_ttest([5, 5, 5, 5], [5, 5, 5])
    assert est.estimate == 0.0
    assert est.statistic == 0.0
    assert est.p_value == 1.0  # no difference, no evidence


def test_power_inputs_are_validated() -> None:
    with pytest.raises(ValueError):
        required_total_n_two_means(2.0, 40.0, ratio=0.0)  # ratio out of (0,1)
    with pytest.raises(ValueError):
        required_total_n_two_means(2.0, 40.0, ratio=1.5)
    with pytest.raises(ValueError):
        required_total_n_two_means(0.0, 40.0)  # mde must be > 0
    with pytest.raises(ValueError):
        power_two_proportions(0.2, 1.5, 1000, 1000)  # p_treatment outside (0,1)
    with pytest.raises(ValueError):
        mde_two_proportions(0.2, 1000, 1000, power=0.04)  # power <= alpha is infeasible


def test_required_n_two_proportions_infeasible_raises() -> None:
    with pytest.raises(ValueError):
        required_total_n_two_proportions(0.9, 0.2)  # p_control + mde = 1.1 > 1


# --------------------------------------------------------------------------- #
# Frequentist behaviour by simulation -- the exact quantities the Phase-5 gates use
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_welch_coverage_and_aa_fpr_by_simulation() -> None:
    rng = np.random.default_rng(7)
    n, sd, effect, alpha, k = 4000, 1.0, 0.06, 0.05, 3000
    cover = aa_reject = 0
    for _ in range(k):
        est = welch_ttest(rng.normal(effect, sd, n), rng.normal(0.0, sd, n), alpha)
        cover += est.ci_contains(effect)
        aa_reject += welch_ttest(rng.normal(0, sd, n), rng.normal(0, sd, n), alpha).significant
    assert abs(cover / k - 0.95) < 0.02
    assert abs(aa_reject / k - alpha) < 0.02


@pytest.mark.slow
def test_two_proportion_coverage_and_aa_fpr_by_simulation() -> None:
    rng = np.random.default_rng(11)
    n, p_c, effect, alpha, k = 4000, 0.20, 0.02, 0.05, 3000
    p_t = p_c + effect
    cover = aa_reject = 0
    for _ in range(k):
        est = two_proportion_ztest(n, rng.binomial(n, p_t), n, rng.binomial(n, p_c), alpha)
        cover += est.ci_contains(effect)
        aa_reject += two_proportion_ztest(
            n, rng.binomial(n, p_c), n, rng.binomial(n, p_c), alpha
        ).significant
    assert cover / k > 0.93  # Agresti-Caffo: at/above nominal coverage
    assert abs(aa_reject / k - alpha) < 0.02
