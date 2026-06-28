"""Two-sample effect estimators with confidence intervals and p-values.

Estimators are implemented *explicitly* (not delegated to a black-box) so the
statistics are visible and auditable, and validated against statsmodels/analytic
truth in the tests:

* **Continuous** metrics -> Welch's t-test (unequal variances): effect = difference
  in means, t-based CI (so coverage is >= nominal), Welch-Satterthwaite dof.
* **Proportion** metrics -> two-proportion z-test: effect = difference in rates,
  Agresti-Caffo CI for the interval (better coverage than Wald, never degenerate),
  pooled-variance z for the p-value (correct size under the null -- this is what the
  A/A gate relies on).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class EffectEstimate:
    """Result of a two-sample comparison (treatment - control)."""

    estimate: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float
    statistic: float
    dof: float | None
    n_treatment: int
    n_control: int
    alpha: float
    method: str

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    @property
    def ci_width(self) -> float:
        return self.ci_high - self.ci_low

    def ci_contains(self, value: float) -> bool:
        """Whether the confidence interval covers ``value`` (used by the coverage gate)."""
        return self.ci_low <= value <= self.ci_high

    def to_dict(self) -> dict:
        return asdict(self)


def welch_ttest(treatment: np.ndarray, control: np.ndarray, alpha: float = 0.05) -> EffectEstimate:
    """Welch's two-sample t-test for a continuous metric (treatment - control)."""
    t = np.asarray(treatment, dtype=float)
    c = np.asarray(control, dtype=float)
    n_t, n_c = t.size, c.size
    if n_t < 2 or n_c < 2:
        raise ValueError("Welch's t-test needs at least 2 observations per group.")

    mean_t, mean_c = float(t.mean()), float(c.mean())
    var_t, var_c = float(t.var(ddof=1)), float(c.var(ddof=1))
    estimate = mean_t - mean_c
    se = float(np.sqrt(var_t / n_t + var_c / n_c))

    if se == 0.0:
        # Both groups are constant (zero variance): the Welch-Satterthwaite dof is
        # 0/0. Match scipy's limiting behaviour instead of crashing: identical means
        # -> no evidence (p=1); separated means -> infinitely significant (p=0).
        if estimate == 0.0:
            t_stat, p_value = 0.0, 1.0
        else:
            t_stat, p_value = (math.inf if estimate > 0 else -math.inf), 0.0
        return EffectEstimate(
            estimate=estimate,
            se=0.0,
            ci_low=estimate,
            ci_high=estimate,
            p_value=p_value,
            statistic=float(t_stat),
            dof=float(n_t + n_c - 2),
            n_treatment=n_t,
            n_control=n_c,
            alpha=alpha,
            method="welch_ttest",
        )

    # Welch-Satterthwaite degrees of freedom (well-defined since se > 0).
    dof = (var_t / n_t + var_c / n_c) ** 2 / (
        (var_t / n_t) ** 2 / (n_t - 1) + (var_c / n_c) ** 2 / (n_c - 1)
    )
    t_stat = estimate / se
    p_value = float(2 * stats.t.sf(abs(t_stat), dof))
    crit = float(stats.t.ppf(1 - alpha / 2, dof))
    return EffectEstimate(
        estimate=estimate,
        se=se,
        ci_low=estimate - crit * se,
        ci_high=estimate + crit * se,
        p_value=p_value,
        statistic=float(t_stat),
        dof=float(dof),
        n_treatment=n_t,
        n_control=n_c,
        alpha=alpha,
        method="welch_ttest",
    )


def two_proportion_ztest(
    n_treatment: int,
    successes_treatment: int,
    n_control: int,
    successes_control: int,
    alpha: float = 0.05,
) -> EffectEstimate:
    """Two-proportion z-test for a binary metric (treatment rate - control rate).

    * **Point estimate**: the raw rate difference p_t - p_c.
    * **CI**: the **Agresti-Caffo** interval (add one success and one failure per arm).
      It has markedly better coverage than the raw Wald interval for a difference of
      proportions and is never degenerate (no zero-width CI when an arm is all-0/all-1).
    * **p-value**: the pooled-variance z-test of H0: p_t == p_c (correct Type-I error
      under the null -- this is what the A/A gate relies on).
    """
    if n_treatment <= 0 or n_control <= 0:
        raise ValueError("Both groups need at least one observation.")

    p_t = successes_treatment / n_treatment
    p_c = successes_control / n_control
    estimate = p_t - p_c
    crit = float(stats.norm.ppf(1 - alpha / 2))

    # Agresti-Caffo confidence interval (centered on the adjusted difference).
    pt_adj = (successes_treatment + 1) / (n_treatment + 2)
    pc_adj = (successes_control + 1) / (n_control + 2)
    se_ac = float(
        np.sqrt(pt_adj * (1 - pt_adj) / (n_treatment + 2) + pc_adj * (1 - pc_adj) / (n_control + 2))
    )
    center_ac = pt_adj - pc_adj

    # Pooled SE for the hypothesis test (correct Type-I error under the null).
    p_pool = (successes_treatment + successes_control) / (n_treatment + n_control)
    se_pooled = float(np.sqrt(p_pool * (1 - p_pool) * (1 / n_treatment + 1 / n_control)))
    z = estimate / se_pooled if se_pooled > 0 else 0.0
    p_value = float(2 * stats.norm.sf(abs(z)))

    return EffectEstimate(
        estimate=estimate,
        se=se_ac,
        ci_low=center_ac - crit * se_ac,
        ci_high=center_ac + crit * se_ac,
        p_value=p_value,
        statistic=float(z),
        dof=None,
        n_treatment=n_treatment,
        n_control=n_control,
        alpha=alpha,
        method="two_proportion_ztest",
    )


def estimate_effect(
    values: np.ndarray, variant: np.ndarray, metric_type: str, alpha: float = 0.05
) -> EffectEstimate:
    """Dispatch to the right two-sample estimator given a per-unit array + variant.

    ``metric_type`` is ``"continuous"`` or ``"proportion"``. For proportions, ``values``
    must be 0/1.
    """
    values = np.asarray(values, dtype=float)
    variant = np.asarray(variant)
    treat = values[variant == 1]
    control = values[variant == 0]

    if metric_type == "continuous":
        return welch_ttest(treat, control, alpha=alpha)
    if metric_type == "proportion":
        binary = np.isin(values, (0.0, 1.0))
        if not binary.all():
            raise ValueError("proportion metric requires values in {0, 1}")
        # Values are exactly 0/1 (validated), so the integer sum is exact.
        return two_proportion_ztest(
            n_treatment=treat.size,
            successes_treatment=int(treat.sum()),
            n_control=control.size,
            successes_control=int(control.sum()),
            alpha=alpha,
        )
    raise ValueError(f"Unknown metric_type: {metric_type!r} (expected continuous|proportion)")
