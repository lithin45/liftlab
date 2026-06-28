"""CUPED, Controlled-experiment Using Pre-Experiment Data (variance reduction).

CUPED (Deng, Xu, Kohavi & Walker, 2013) reduces the variance of a metric using a
pre-period covariate ``X`` that is correlated with the outcome ``Y`` but unaffected by
the treatment. Define the adjusted metric

    Y_cuped = Y - theta * (X - mean(X)),   theta = Cov(Y, X) / Var(X).

Because ``X`` is pre-treatment (independent of assignment), ``theta`` can be estimated
on the pooled sample without biasing the treatment effect: the difference in adjusted
means equals the difference in raw means minus ``theta * (Xbar_T - Xbar_C)``, and the
covariate is balanced in expectation, so the ATE is preserved. The variance, however,
shrinks by a factor ``rho^2`` where ``rho = corr(Y, X)``:

    Var(Y_cuped) = Var(Y) * (1 - rho^2).

So a covariate with ``rho ~= 0.6`` yields ``~36%`` variance reduction -- a tighter CI
for the *same* experiment. We report the achieved reduction and the adjusted estimate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from liftlab.estimators.twosample import EffectEstimate, welch_ttest


@dataclass
class CupedResult:
    """CUPED outcome: theta, achieved variance reduction, and adjusted vs. naive estimates."""

    theta: float
    correlation: float
    variance_reduction: float  # 1 - Var(Y_cuped)/Var(Y); equals rho^2 asymptotically
    unadjusted: EffectEstimate
    adjusted: EffectEstimate

    @property
    def se_reduction(self) -> float:
        """Fractional reduction in the estimator's standard error."""
        if self.unadjusted.se == 0:
            return 0.0
        return 1.0 - self.adjusted.se / self.unadjusted.se

    def to_dict(self) -> dict:
        d = asdict(self)
        d["se_reduction"] = self.se_reduction
        return d


def cuped_theta(y: np.ndarray, covariate: np.ndarray) -> float:
    """Optimal CUPED coefficient theta = Cov(Y, X) / Var(X) (0 if X is constant)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(covariate, dtype=float)
    var_x = float(np.var(x, ddof=1))
    if var_x == 0.0:
        return 0.0
    return float(np.cov(y, x, ddof=1)[0, 1] / var_x)


def cuped_adjust(
    y: np.ndarray, covariate: np.ndarray, theta: float | None = None
) -> tuple[np.ndarray, float]:
    """Return (Y_cuped, theta). If ``theta`` is None it is estimated from (y, covariate)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(covariate, dtype=float)
    if theta is None:
        theta = cuped_theta(y, x)
    return y - theta * (x - x.mean()), theta


def cuped_estimate(
    y: np.ndarray, covariate: np.ndarray, variant: np.ndarray, alpha: float = 0.05
) -> CupedResult:
    """Estimate the treatment effect with and without CUPED, and the variance reduction.

    ``theta`` is estimated on the pooled sample (valid because the covariate is
    pre-treatment). Both estimates use Welch's t-test on the respective arrays.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(covariate, dtype=float)
    variant = np.asarray(variant)

    theta = cuped_theta(y, x)
    y_cuped = y - theta * (x - x.mean())

    unadjusted = welch_ttest(y[variant == 1], y[variant == 0], alpha)
    adjusted = welch_ttest(y_cuped[variant == 1], y_cuped[variant == 0], alpha)

    var_y = float(np.var(y, ddof=1))
    variance_reduction = 1.0 - float(np.var(y_cuped, ddof=1)) / var_y if var_y > 0 else 0.0
    correlation = float(np.corrcoef(y, x)[0, 1]) if np.var(x, ddof=1) > 0 else 0.0

    return CupedResult(
        theta=theta,
        correlation=correlation,
        variance_reduction=variance_reduction,
        unadjusted=unadjusted,
        adjusted=adjusted,
    )
