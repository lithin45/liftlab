"""Power / Minimum-Detectable-Effect (MDE) calculators.

Closed-form normal-approximation power for the two estimators, plus the inverse
(required N for a target MDE). Continuous formulas have closed-form inverses;
proportion inverses are solved numerically (the alternative-hypothesis variance
depends on the effect being detected).

Inputs are validated and infeasible targets raise a clear ``ValueError`` rather than
returning silent garbage or surfacing SciPy's cryptic bracket error.

All functions are validated in the tests against statsmodels (continuous) and a
Monte-Carlo rejection-rate simulation (proportions).
"""

from __future__ import annotations

import math
from collections.abc import Callable

from scipy import stats
from scipy.optimize import brentq


def _z(p: float) -> float:
    return float(stats.norm.ppf(p))


def _require_prob(name: str, x: float) -> None:
    if not 0.0 < x < 1.0:
        raise ValueError(f"{name} must be in (0, 1), got {x!r}")


def _require_positive(name: str, x: float) -> None:
    if not x > 0.0:
        raise ValueError(f"{name} must be > 0, got {x!r}")


def _require_design(alpha: float, power: float | None = None, ratio: float | None = None) -> None:
    _require_prob("alpha", alpha)
    if power is not None:
        _require_prob("power", power)
        if power <= alpha:
            raise ValueError(f"power ({power}) must exceed alpha ({alpha}) to be detectable")
    if ratio is not None:
        _require_prob("ratio (treatment fraction)", ratio)


def _solve(gap: Callable[[float], float], lo: float, hi: float, what: str) -> float:
    """Bracketed root solve with a meaningful error when the target is infeasible."""
    flo, fhi = gap(lo), gap(hi)
    if flo == 0.0:
        return float(lo)
    if fhi == 0.0:
        return float(hi)
    if (flo < 0.0) == (fhi < 0.0):
        raise ValueError(
            f"cannot solve for {what}: the target is infeasible for this design "
            f"(no root in [{lo:g}, {hi:g}])."
        )
    return float(brentq(gap, lo, hi))


# --------------------------------------------------------------------------- #
# Continuous (difference in means)
# --------------------------------------------------------------------------- #
def _se_two_means(sd: float, n_treatment: float, n_control: float) -> float:
    return sd * math.sqrt(1.0 / n_treatment + 1.0 / n_control)


def power_two_means(
    effect: float, sd: float, n_treatment: float, n_control: float, alpha: float = 0.05
) -> float:
    """Two-sided power to detect ``effect`` given a per-unit sd and group sizes."""
    _require_positive("sd", sd)
    _require_positive("n_treatment", n_treatment)
    _require_positive("n_control", n_control)
    _require_prob("alpha", alpha)
    se = _se_two_means(sd, n_treatment, n_control)
    zc = _z(1 - alpha / 2)
    ncp = abs(effect) / se
    return float(stats.norm.cdf(ncp - zc) + stats.norm.cdf(-ncp - zc))


def mde_two_means(
    sd: float, n_treatment: float, n_control: float, alpha: float = 0.05, power: float = 0.80
) -> float:
    """Minimum detectable effect (absolute) for the given design."""
    _require_positive("sd", sd)
    _require_positive("n_treatment", n_treatment)
    _require_positive("n_control", n_control)
    _require_design(alpha, power)
    se = _se_two_means(sd, n_treatment, n_control)
    return float((_z(1 - alpha / 2) + _z(power)) * se)


def required_total_n_two_means(
    mde: float, sd: float, alpha: float = 0.05, power: float = 0.80, ratio: float = 0.5
) -> float:
    """Total N (both arms) to detect ``mde``. ``ratio`` is the treatment fraction."""
    _require_positive("mde", abs(mde))
    _require_positive("sd", sd)
    _require_design(alpha, power, ratio)
    allocation = 1.0 / ratio + 1.0 / (1.0 - ratio)
    return float(((_z(1 - alpha / 2) + _z(power)) * sd / mde) ** 2 * allocation)


# --------------------------------------------------------------------------- #
# Proportions (difference in rates)
# --------------------------------------------------------------------------- #
def power_two_proportions(
    p_control: float,
    p_treatment: float,
    n_treatment: float,
    n_control: float,
    alpha: float = 0.05,
) -> float:
    """Two-sided power for a two-proportion z-test.

    Uses the pooled variance under H0 (matching the test statistic) and the unpooled
    variance under H1, which is the standard two-proportion power formula.
    """
    _require_prob("p_control", p_control)
    _require_prob("p_treatment", p_treatment)
    _require_positive("n_treatment", n_treatment)
    _require_positive("n_control", n_control)
    _require_prob("alpha", alpha)

    effect = abs(p_treatment - p_control)
    p_bar = (p_treatment * n_treatment + p_control * n_control) / (n_treatment + n_control)
    se0 = math.sqrt(p_bar * (1 - p_bar) * (1.0 / n_treatment + 1.0 / n_control))
    se1 = math.sqrt(
        p_treatment * (1 - p_treatment) / n_treatment + p_control * (1 - p_control) / n_control
    )
    zc = _z(1 - alpha / 2)
    return float(
        stats.norm.cdf((effect - zc * se0) / se1) + stats.norm.cdf((-effect - zc * se0) / se1)
    )


def mde_two_proportions(
    p_control: float,
    n_treatment: float,
    n_control: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Minimum detectable absolute lift over ``p_control`` for the given design."""
    _require_prob("p_control", p_control)
    _require_positive("n_treatment", n_treatment)
    _require_positive("n_control", n_control)
    _require_design(alpha, power)

    def gap(delta: float) -> float:
        return (
            power_two_proportions(p_control, p_control + delta, n_treatment, n_control, alpha)
            - power
        )

    hi = (1.0 - p_control) - 1e-9
    return _solve(gap, 1e-9, hi, "minimum detectable lift")


def required_total_n_two_proportions(
    mde: float,
    p_control: float,
    alpha: float = 0.05,
    power: float = 0.80,
    ratio: float = 0.5,
) -> float:
    """Total N (both arms) to detect an absolute lift ``mde`` over ``p_control``."""
    _require_prob("p_control", p_control)
    _require_positive("mde", mde)
    _require_design(alpha, power, ratio)
    if not 0.0 < p_control + mde < 1.0:
        raise ValueError(f"p_control + mde = {p_control + mde} is outside (0, 1)")

    def gap(total_n: float) -> float:
        n_t = ratio * total_n
        n_c = (1.0 - ratio) * total_n
        return power_two_proportions(p_control, p_control + mde, n_t, n_c, alpha) - power

    return _solve(gap, 4.0, 1e9, "required total N")
