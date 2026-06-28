"""Observational causal estimators for the broken-randomization fallback.

The estimators are implemented transparently (consistent with LiftLab's "don't hide it
in a black box" ethos), and DoWhy is used to (a) cross-validate the IPW point estimate
and (b) provide refutation/sensitivity tests:

* **Naive**, difference in post-period means. BIASED under confounding (shown for contrast).
* **Difference-in-Differences (DiD)**, difference of the within-unit pre->post change
    between arms. Identifies the ATE under the **parallel-trends** assumption (the
    confounder's effect is time-invariant, so it cancels in the differencing).
* **Inverse-Propensity Weighting (IPW)**, fit P(T|confounder) by logistic regression and
    reweight (Hajek estimator) to balance the arms. Identifies the ATE under
    **unconfoundedness** (no unobserved confounders) and **overlap** (0<P(T|x)<1). CI by
    nonparametric bootstrap.

Sensitivity: a placebo treatment (permuted) should yield ~0; DoWhy's random-common-cause
and data-subset refuters should leave the estimate stable.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

from liftlab.estimators.twosample import EffectEstimate, welch_ttest


@dataclass
class IPWEstimate:
    estimate: float
    ci_low: float
    ci_high: float
    se: float
    n_boot: int
    min_propensity: float
    max_propensity: float

    @property
    def ci_width(self) -> float:
        return self.ci_high - self.ci_low

    def ci_contains(self, value: float) -> bool:
        return self.ci_low <= value <= self.ci_high

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CausalResult:
    """Bundle of the naive + causal estimates on a confounded scenario, vs the known truth."""

    true_effect: float
    naive: EffectEstimate
    did: EffectEstimate
    ipw: IPWEstimate
    dowhy_ipw: float | None
    placebo_effect: float
    refutations: dict

    def to_dict(self) -> dict:
        return {
            "true_effect": self.true_effect,
            "naive": self.naive.to_dict(),
            "did": self.did.to_dict(),
            "ipw": self.ipw.to_dict(),
            "dowhy_ipw": self.dowhy_ipw,
            "placebo_effect": self.placebo_effect,
            "refutations": self.refutations,
        }


def naive_estimate(data: pd.DataFrame, alpha: float = 0.05) -> EffectEstimate:
    """Difference in post-period means (biased under confounding)."""
    y = data["y_post"].to_numpy()
    t = data["treatment"].to_numpy()
    return welch_ttest(y[t == 1], y[t == 0], alpha=alpha)


def did_estimate(data: pd.DataFrame, alpha: float = 0.05) -> EffectEstimate:
    """Difference-in-differences via the within-unit pre->post change."""
    delta = (data["y_post"] - data["y_pre"]).to_numpy()
    t = data["treatment"].to_numpy()
    return welch_ttest(delta[t == 1], delta[t == 0], alpha=alpha)


def _propensity(z: np.ndarray, t: np.ndarray, trim: float = 0.02) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=1000)
    model.fit(z.reshape(-1, 1), t)
    ps = model.predict_proba(z.reshape(-1, 1))[:, 1]
    return np.clip(ps, trim, 1.0 - trim)


def _hajek_ate(z: np.ndarray, t: np.ndarray, y: np.ndarray) -> float:
    """Stabilized (Hajek) inverse-propensity-weighted ATE."""
    ps = _propensity(z, t)
    w1 = t / ps
    w0 = (1 - t) / (1 - ps)
    return float(np.sum(w1 * y) / np.sum(w1) - np.sum(w0 * y) / np.sum(w0))


def ipw_estimate(
    data: pd.DataFrame, alpha: float = 0.05, n_boot: int = 300, seed: int = 0
) -> IPWEstimate:
    """IPW (Hajek) ATE with a nonparametric bootstrap confidence interval."""
    z = data["confounder"].to_numpy()
    t = data["treatment"].to_numpy()
    y = data["y_post"].to_numpy()

    estimate = _hajek_ate(z, t, y)
    ps = _propensity(z, t)

    rng = np.random.default_rng(seed)
    n = y.size
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = _hajek_ate(z[idx], t[idx], y[idx])
    # Normal-approximation bootstrap CI (estimate ± z * bootstrap SE). Better-calibrated
    # than the percentile interval for IPW's mildly-skewed bootstrap distribution.
    se = float(boots.std(ddof=1))
    crit = float(stats.norm.ppf(1 - alpha / 2))

    return IPWEstimate(
        estimate=estimate,
        ci_low=estimate - crit * se,
        ci_high=estimate + crit * se,
        se=se,
        n_boot=n_boot,
        min_propensity=float(ps.min()),
        max_propensity=float(ps.max()),
    )


def placebo_effect(data: pd.DataFrame, seed: int = 0, n_permutations: int = 25) -> float:
    """Mean IPW estimate under permuted (placebo) treatments, ~0 if the method is valid.

    Averaging over permutations is deliberate: a single permutation has SD ~0.4, so one
    draw is a noisy validity check; the mean over ``n_permutations`` is a stable ~0.
    """
    rng = np.random.default_rng(seed)
    z = data["confounder"].to_numpy()
    y = data["y_post"].to_numpy()
    t = data["treatment"].to_numpy()
    effects = [_hajek_ate(z, rng.permutation(t), y) for _ in range(n_permutations)]
    return float(np.mean(effects))


def dowhy_analysis(
    data: pd.DataFrame, num_simulations: int = 20, random_seed: int = 0
) -> tuple[float | None, dict]:
    """Cross-validate IPW with DoWhy and run its refutation/sensitivity tests.

    Returns ``(dowhy_ipw_value, refutations)``. ``random_seed`` is passed to the refuters
    so their values are reproducible. Degrades gracefully (returns ``None`` / empty) if
    DoWhy is unavailable so the rest of the report still renders.
    """
    logging.getLogger("dowhy").setLevel(logging.ERROR)
    try:
        from dowhy import CausalModel
    except ImportError:
        return None, {}

    model = CausalModel(
        data=data, treatment="treatment", outcome="y_post", common_causes=["confounder"]
    )
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        identified, method_name="backdoor.propensity_score_weighting", target_units="ate"
    )

    refutations: dict = {}
    for label, method in (
        ("random_common_cause", "random_common_cause"),
        ("data_subset", "data_subset_refuter"),
    ):
        try:
            ref = model.refute_estimate(
                identified,
                estimate,
                method_name=method,
                num_simulations=num_simulations,
                random_seed=random_seed,
            )
            refutations[label] = float(ref.new_effect)
        except Exception:
            refutations[label] = None

    return float(estimate.value), refutations


def run_causal_analysis(
    data: pd.DataFrame,
    true_effect: float,
    *,
    alpha: float = 0.05,
    n_boot: int = 300,
    seed: int = 0,
    with_dowhy: bool = True,
) -> CausalResult:
    """Run naive + DiD + IPW on a confounded scenario, plus DoWhy cross-check + refuters."""
    naive = naive_estimate(data, alpha)
    did = did_estimate(data, alpha)
    ipw = ipw_estimate(data, alpha, n_boot=n_boot, seed=seed)
    placebo = placebo_effect(data, seed=seed)
    dowhy_ipw, refutations = dowhy_analysis(data, random_seed=seed) if with_dowhy else (None, {})
    return CausalResult(
        true_effect=true_effect,
        naive=naive,
        did=did,
        ipw=ipw,
        dowhy_ipw=dowhy_ipw,
        placebo_effect=placebo,
        refutations=refutations,
    )
