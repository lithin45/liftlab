"""A deliberately-CONFOUNDED experiment, the setting where randomization has broken.

DISCLOSURE: like the rest of LiftLab, the true effect is INJECTED and known. Here we
also break randomization on purpose: a unit's treatment probability depends on its
pre-period spend (the confounder z), so the treatment and control groups are no longer
comparable. The naive post-period difference is therefore BIASED, and we use it to show
that the observational causal fallbacks (difference-in-differences, inverse-propensity
weighting) recover the known truth while the naive estimate does not.

Generative model (a two-period panel):
    z          ~ standardized pre-period covariate (the confounder)
    P(T=1|z)   = sigmoid(gamma * z)                 # confounded assignment
    y_pre      = base + lambda*z + eps_pre          # pre-period outcome
    y_post     = base + trend + lambda*z + tau*T + eps_post

Because the confounder's effect (lambda*z) is the SAME in both periods, differencing
(y_post - y_pre) removes it -> DiD identifies tau under parallel trends. The naive
post-only difference carries the bias lambda*(zbar_T - zbar_C) > 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from liftlab.config import Config
from liftlab.simulation.simulate import standardize_covariate


@dataclass
class CausalScenario:
    """A confounded two-period panel with a known true effect."""

    data: pd.DataFrame  # columns: unit_id, confounder, treatment, y_pre, y_post
    true_effect: float
    confounding_strength: float
    naive_bias_direction: str  # "up" | "down"

    @property
    def n_treatment(self) -> int:
        return int(self.data["treatment"].sum())


def make_confounded_scenario(
    config: Config,
    covariate_values: np.ndarray,
    *,
    seed: int,
    true_effect: float | None = None,
    confounding: float = 1.0,
    confounder_outcome_coef: float = 6.0,
    trend: float = 5.0,
    confounder_clip: float = 3.0,
) -> CausalScenario:
    """Generate a confounded two-period panel from population covariates.

    Parameters
    ----------
    confounding:
        Strength of the confounder's effect on assignment (logit slope). 0 -> random.
    confounder_outcome_coef:
        ``lambda``, the confounder's effect on the outcome level.
    trend:
        Common pre->post time shift (equal for both arms: parallel trends hold).
    confounder_clip:
        The standardized confounder is **winsorized** to [-clip, +clip]. The real
        pre-period covariate is extremely right-skewed (z up to +19); without this, the
        logistic propensity P(T|z)=sigmoid(gamma*z) would reach 1.0 for tail units, an
        overlap violation that biases IPW. Winsorizing keeps propensities bounded
        (sigmoid(±3) ≈ [0.05, 0.95]) so overlap holds and IPW is unbiased, while the
        confounding (and thus the naive bias) remains strong.
    """
    rng = np.random.default_rng(seed)
    z = standardize_covariate(np.asarray(covariate_values, dtype=float))
    z = np.clip(z, -confounder_clip, confounder_clip)
    n = z.size

    rev = config.experiment.metrics.revenue
    tau = rev.true_effect_absolute if true_effect is None else true_effect
    base = rev.baseline_mean
    lam = confounder_outcome_coef
    rho = config.experiment.covariate.target_correlation
    noise_sd = math.sqrt(max(0.0, 1.0 - rho**2)) * rev.outcome_sd

    # Confounded assignment: higher pre-period spend -> more likely treated.
    p_treat = 1.0 / (1.0 + np.exp(-confounding * z))
    treatment = (rng.random(n) < p_treat).astype(int)

    y_pre = base + lam * z + rng.normal(0.0, noise_sd, n)
    y_post = base + trend + lam * z + tau * treatment + rng.normal(0.0, noise_sd, n)

    data = pd.DataFrame(
        {
            "unit_id": np.arange(n, dtype=np.int64),
            "confounder": z,
            "treatment": treatment,
            "y_pre": y_pre,
            "y_post": y_post,
        }
    )
    return CausalScenario(
        data=data,
        true_effect=float(tau),
        confounding_strength=float(confounding),
        naive_bias_direction="up" if lam > 0 else "down",
    )
