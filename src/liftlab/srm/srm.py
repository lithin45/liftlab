"""Sample-Ratio-Mismatch (SRM) detection.

A chi-square goodness-of-fit test comparing the *realized* assignment counts to the
*intended* split. A significant result means the randomization is broken (logging loss,
bucketing bug, redirect leakage, ...) and the experiment's results are untrustworthy,
a critical guardrail before believing any lift.

The test uses a deliberately *strict* threshold (default p < 0.001): an SRM alarm
should fire only on a genuine imbalance, since a false alarm needlessly discards a
valid experiment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from scipy import stats


@dataclass
class SRMResult:
    n_control: int
    n_treatment: int
    expected_ratio: float  # intended P(treatment)
    observed_ratio: float
    chi_square: float
    p_value: float
    threshold: float
    is_srm: bool  # True -> assignment is broken

    def to_dict(self) -> dict:
        return asdict(self)


def srm_check(
    n_control: int,
    n_treatment: int,
    expected_ratio: float = 0.5,
    threshold: float = 0.001,
) -> SRMResult:
    """Chi-square SRM test of realized vs. intended assignment.

    Parameters
    ----------
    n_control, n_treatment:
        Realized unit counts per arm.
    expected_ratio:
        Intended P(unit -> treatment).
    threshold:
        Flag an SRM when the chi-square p-value is below this (default 0.001).
    """
    if not 0.0 < expected_ratio < 1.0:
        raise ValueError(f"expected_ratio must be in (0, 1), got {expected_ratio!r}")
    total = n_control + n_treatment
    if total <= 0:
        raise ValueError("need at least one assigned unit")

    expected_treatment = total * expected_ratio
    expected_control = total * (1.0 - expected_ratio)
    chi_square = (n_treatment - expected_treatment) ** 2 / expected_treatment + (
        n_control - expected_control
    ) ** 2 / expected_control
    p_value = float(stats.chi2.sf(chi_square, df=1))

    return SRMResult(
        n_control=int(n_control),
        n_treatment=int(n_treatment),
        expected_ratio=float(expected_ratio),
        observed_ratio=float(n_treatment / total),
        chi_square=float(chi_square),
        p_value=p_value,
        threshold=float(threshold),
        is_srm=p_value < threshold,
    )
