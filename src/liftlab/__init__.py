"""LiftLab, a locally-runnable A/B testing & experimentation engine.

LiftLab demonstrates a *provably correct* experimentation stack: power/MDE,
two-sample tests, CUPED variance reduction, sample-ratio-mismatch detection,
and observational causal fallbacks. Because the treatment effect is synthetic
and injected by us, the true effect is known and every estimator is validated
by recovering that truth across many Monte-Carlo runs.

DISCLOSURE: this is not a real-world experiment. See ``config/experiment.yaml``
and the README for the full synthetic-design disclosure.
"""

__version__ = "0.1.0"
