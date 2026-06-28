"""Phase 5: the Monte-Carlo validation harness enforces all four gates."""

from __future__ import annotations

import math

import numpy as np
import pytest

from liftlab.config import load_config
from liftlab.simulation.validation import (
    GateRow,
    ValidationReport,
    coverage_lower_bound,
    format_report,
    run_validation,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_coverage_lower_bound_is_mc_noise_aware() -> None:
    lb = coverage_lower_bound(0.94, 2.33, 3000)
    assert lb == pytest.approx(0.94 - 2.33 * math.sqrt(0.94 * 0.06 / 3000), rel=1e-12)
    # Below the 0.94 floor, so a correct estimator (true coverage ~0.945-0.95) passes...
    assert lb < 0.94
    # ...but not so low it would accept a genuinely broken estimator (e.g. <0.92).
    assert lb > 0.92


def test_report_aggregation_requires_every_gate() -> None:
    good = GateRow("x", 0.95, True, 0.05, True)
    bad_cov = GateRow("y", 0.90, False, 0.05, True)
    base = {
        "n_simulations": 100,
        "sample_size": 1000,
        "alpha": 0.05,
        "coverage_target": 0.95,
        "coverage_floor": 0.94,
        "coverage_lower_bound": 0.93,
        "cuped_variance_reduction": 0.36,
        "cuped_threshold": 0.30,
        "cuped_pass": True,
        "srm_imbalanced_detected": True,
        "srm_balanced_cleared": True,
        "srm_pass": True,
        "config_hash": "abc",
    }
    bad_fpr = GateRow("y", 0.95, True, 0.20, False)
    assert ValidationReport(estimators=[good], **base).all_passed
    assert not ValidationReport(estimators=[bad_cov], **base).all_passed
    assert not ValidationReport(estimators=[bad_fpr], **base).all_passed
    assert not ValidationReport(estimators=[good], **{**base, "cuped_pass": False}).all_passed
    assert not ValidationReport(estimators=[good], **{**base, "srm_pass": False}).all_passed


def _report(**overrides) -> ValidationReport:
    base = {
        "n_simulations": 10,
        "sample_size": 100,
        "alpha": 0.05,
        "coverage_target": 0.95,
        "coverage_floor": 0.94,
        "coverage_lower_bound": 0.93,
        "estimators": [GateRow("x", 0.95, True, 0.05, True)],
        "cuped_variance_reduction": 0.36,
        "cuped_threshold": 0.30,
        "cuped_pass": True,
        "srm_imbalanced_detected": True,
        "srm_balanced_cleared": True,
        "srm_pass": True,
        "config_hash": "abc",
    }
    base.update(overrides)
    return ValidationReport(**base)


def test_cli_eval_exits_nonzero_on_gate_miss(monkeypatch) -> None:
    """Hard requirement: `liftlab eval` must exit non-zero if any gate is missed."""
    from liftlab import cli
    from liftlab.simulation import validation as v

    monkeypatch.setattr(v, "run_validation", lambda *a, **k: _report())
    assert cli.main(["eval"]) == 0

    monkeypatch.setattr(v, "run_validation", lambda *a, **k: _report(cuped_pass=False))
    assert cli.main(["eval"]) == 1


@pytest.mark.slow
def test_all_validation_gates_pass(cfg) -> None:
    """THE eval gate: coverage >=95%, A/A FPR ~5%, CUPED >=30%, SRM detection."""
    rng = np.random.default_rng(20240601)
    covariate = rng.lognormal(mean=3.5, sigma=0.8, size=5000)
    report = run_validation(cfg, covariate_values=covariate, n_simulations=1500)

    assert report.all_passed, format_report(report)

    # Each estimator clears the actual gate (track the harness's own bound, not a
    # hand-picked constant) and lands near nominal with ~5% A/A FPR.
    assert len(report.estimators) == 3
    for row in report.estimators:
        assert row.coverage_pass and row.fpr_pass
        assert row.coverage >= report.coverage_lower_bound
        assert row.coverage < 0.98  # loose sanity: not vacuously wide CIs
        assert abs(row.fpr - cfg.power.alpha) < 0.02

    assert report.cuped_variance_reduction >= cfg.validation.cuped_min_variance_reduction
    assert report.srm_imbalanced_detected
    assert report.srm_balanced_cleared


@pytest.mark.slow
def test_coverage_gate_catches_a_broken_estimator(cfg) -> None:
    """The gate's whole purpose: a deliberately over-confident (too-narrow) CI must give
    coverage well below the floor, so the coverage gate would FAIL it."""
    from liftlab.estimators.twosample import welch_ttest
    from liftlab.simulation.simulate import draw_experiment, standardize_covariate

    rng = np.random.default_rng(5)
    z = standardize_covariate(rng.lognormal(mean=3.5, sigma=0.8, size=5000))
    truth = cfg.experiment.metrics.revenue.true_effect_absolute
    k = 800
    broken = 0
    for i in range(k):
        d = draw_experiment(cfg, z, seed=500 + i)
        est = welch_ttest(d.y_revenue[d.variant == 1], d.y_revenue[d.variant == 0])
        half = (est.ci_high - est.ci_low) / 2 * 0.5  # halve the CI -> over-confident
        broken += (est.estimate - half) <= truth <= (est.estimate + half)
    broken_coverage = broken / k
    lb = coverage_lower_bound(cfg.validation.coverage_floor, cfg.validation.coverage_mc_sigma, k)
    assert broken_coverage < lb  # would FAIL the coverage gate
    assert broken_coverage < 0.85  # and is clearly, grossly under-covering


@pytest.mark.slow
def test_coverage_metric_is_truth_sensitive(cfg) -> None:
    """Coverage genuinely discriminates: CIs cover the real truth ~95% of the time but
    almost never cover a clearly-wrong truth several SE away -- so a biased generative
    model (or a broken estimator) WOULD fail the coverage gate, not pass vacuously."""
    from liftlab.estimators.twosample import welch_ttest
    from liftlab.simulation.simulate import draw_experiment, standardize_covariate

    rng = np.random.default_rng(2)
    z = standardize_covariate(rng.lognormal(mean=3.5, sigma=0.8, size=8000))
    truth = cfg.experiment.metrics.revenue.true_effect_absolute
    wrong = truth + 4.0  # several SE away at this N
    k = 500
    covers_truth = covers_wrong = 0
    for i in range(k):
        d = draw_experiment(cfg, z, seed=300 + i)
        est = welch_ttest(d.y_revenue[d.variant == 1], d.y_revenue[d.variant == 0])
        covers_truth += est.ci_contains(truth)
        covers_wrong += est.ci_contains(wrong)
    assert covers_truth / k > 0.90
    assert covers_wrong / k < 0.10
