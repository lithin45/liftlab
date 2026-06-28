"""Observational causal fallback: difference-in-differences / propensity scores (Phase 6)."""

from liftlab.causal.estimators import (
    CausalResult,
    IPWEstimate,
    did_estimate,
    dowhy_analysis,
    ipw_estimate,
    naive_estimate,
    placebo_effect,
    run_causal_analysis,
)
from liftlab.causal.scenario import CausalScenario, make_confounded_scenario

__all__ = [
    "CausalResult",
    "CausalScenario",
    "IPWEstimate",
    "did_estimate",
    "dowhy_analysis",
    "ipw_estimate",
    "make_confounded_scenario",
    "naive_estimate",
    "placebo_effect",
    "run_causal_analysis",
]
