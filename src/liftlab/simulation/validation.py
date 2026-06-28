"""Monte-Carlo validation harness, the heart of LiftLab.

Runs the engine across many synthetic experiments with KNOWN injected effects and
checks that every estimator recovers the truth:

* **Coverage**, each estimator's 95% CI contains the (effective) true effect. A
  correctly-calibrated estimator's true coverage sits *at or just below* nominal at
  finite N, so a hard >=0.95 point threshold is statistically unpassable. The gate
  requires coverage above a finite-sample FLOOR (nominal minus a ~1% allowance) with an
  MC-noise margin: pass if observed >= floor - coverage_mc_sigma * sqrt(floor(1-floor)/M).
* **A/A false-positive rate**, under a no-effect simulation, each estimator rejects
  at ~alpha (within tolerance), correct Type-I control.
* **CUPED variance reduction**, mean reduction across replicates >= the threshold.
* **SRM**, flags an intentionally-imbalanced split and clears a balanced one.

Monte-Carlo replicates hold the units (and covariate) FIXED and re-randomize the
assignment + redraw the outcome noise (seed = base + i), valid frequentist replication
of the experiment's own randomness, conditional on the realized population.

``liftlab eval`` runs this and exits non-zero if any gate is missed (wired into CI).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from liftlab.config import Config, load_config
from liftlab.cuped.cuped import cuped_estimate
from liftlab.estimators.twosample import two_proportion_ztest
from liftlab.simulation.simulate import (
    conversion_effective_ate,
    draw_experiment,
    load_covariate_values,
    sample_covariate,
    standardize_covariate,
)
from liftlab.srm.srm import srm_check

# Disjoint seed offsets so H1 and A/A replicate streams never overlap.
_H1_OFFSET = 1
_AA_OFFSET = 5_000_000


@dataclass
class GateRow:
    """Per-estimator coverage + A/A FPR with their pass/fail verdicts."""

    name: str
    coverage: float
    coverage_pass: bool
    fpr: float
    fpr_pass: bool


@dataclass
class ValidationReport:
    n_simulations: int
    sample_size: int
    alpha: float
    coverage_target: float
    coverage_floor: float
    coverage_lower_bound: float
    estimators: list[GateRow]
    cuped_variance_reduction: float
    cuped_threshold: float
    cuped_pass: bool
    srm_imbalanced_detected: bool
    srm_balanced_cleared: bool
    srm_pass: bool
    config_hash: str

    @property
    def all_passed(self) -> bool:
        return (
            all(r.coverage_pass and r.fpr_pass for r in self.estimators)
            and self.cuped_pass
            and self.srm_pass
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["all_passed"] = self.all_passed
        return d


def coverage_lower_bound(reference: float, mc_sigma: float, m: int) -> float:
    """MC-noise-aware acceptance bound for a coverage estimate from M replicates.

    ``reference`` is the finite-sample coverage floor (not the nominal target).
    """
    return reference - mc_sigma * math.sqrt(reference * (1 - reference) / m)


def _counts(y: np.ndarray, variant: np.ndarray) -> tuple[int, int, int, int]:
    treat = variant == 1
    return (
        int(treat.sum()),
        int(y[treat].sum()),
        int((~treat).sum()),
        int(y[~treat].sum()),
    )


def run_validation(
    config: Config | None = None,
    *,
    covariate_values: np.ndarray | None = None,
    db_path: Path | None = None,
    n_simulations: int | None = None,
    base_seed: int | None = None,
) -> ValidationReport:
    """Run the Monte-Carlo validation and evaluate all gates."""
    config = config or load_config()
    val = config.validation
    alpha = config.power.alpha
    m = int(n_simulations if n_simulations is not None else val.n_simulations)

    if covariate_values is None:
        population = load_covariate_values(db_path)
        covariate_values = sample_covariate(
            np.asarray(population, dtype=float), config.experiment.sample_size, config.seed
        )
    z = standardize_covariate(np.asarray(covariate_values, dtype=float))
    n = int(z.size)

    rev_truth = config.experiment.metrics.revenue.true_effect_absolute
    conv_truth = conversion_effective_ate(config, z)

    base = config.seed if base_seed is None else base_seed
    keys = ("revenue_naive", "revenue_cuped", "conversion_naive")
    cover = dict.fromkeys(keys, 0)
    reject = dict.fromkeys(keys, 0)
    var_reduction_sum = 0.0

    for i in range(m):
        # --- H1 replicate: coverage of the true effect + CUPED variance reduction ---
        d = draw_experiment(config, z, seed=base + _H1_OFFSET + i)
        cup = cuped_estimate(d.y_revenue, z, d.variant, alpha)
        conv = two_proportion_ztest(*_counts(d.y_conversion, d.variant), alpha)
        cover["revenue_naive"] += cup.unadjusted.ci_contains(rev_truth)
        cover["revenue_cuped"] += cup.adjusted.ci_contains(rev_truth)
        cover["conversion_naive"] += conv.ci_contains(conv_truth)
        var_reduction_sum += cup.variance_reduction

        # --- A/A replicate: false-positive rate (no injected effect) ---
        a = draw_experiment(
            config, z, seed=base + _AA_OFFSET + i, revenue_effect=0.0, conversion_lift=0.0
        )
        a_cup = cuped_estimate(a.y_revenue, z, a.variant, alpha)
        a_conv = two_proportion_ztest(*_counts(a.y_conversion, a.variant), alpha)
        reject["revenue_naive"] += a_cup.unadjusted.significant
        reject["revenue_cuped"] += a_cup.adjusted.significant
        reject["conversion_naive"] += a_conv.significant

    lower = coverage_lower_bound(val.coverage_floor, val.coverage_mc_sigma, m)
    labels = {
        "revenue_naive": "revenue (naive)",
        "revenue_cuped": "revenue (CUPED)",
        "conversion_naive": "conversion (naive)",
    }
    rows = []
    for key in keys:
        coverage = cover[key] / m
        fpr = reject[key] / m
        rows.append(
            GateRow(
                name=labels[key],
                coverage=coverage,
                coverage_pass=coverage >= lower,
                fpr=fpr,
                fpr_pass=abs(fpr - alpha) <= val.aa_fpr_tolerance,
            )
        )

    cuped_vr = var_reduction_sum / m
    cuped_pass = cuped_vr >= val.cuped_min_variance_reduction

    # --- SRM: an intentionally-imbalanced split must be flagged; a balanced one cleared ---
    imb = draw_experiment(config, z, seed=base, assignment_ratio=val.srm_imbalance_ratio)
    bal = draw_experiment(config, z, seed=base)
    srm_imb = srm_check(
        int((imb.variant == 0).sum()),
        int((imb.variant == 1).sum()),
        config.experiment.assignment_ratio,
        val.srm_alpha,
    )
    srm_bal = srm_check(
        int((bal.variant == 0).sum()),
        int((bal.variant == 1).sum()),
        config.experiment.assignment_ratio,
        val.srm_alpha,
    )

    return ValidationReport(
        n_simulations=m,
        sample_size=n,
        alpha=alpha,
        coverage_target=val.coverage_target,
        coverage_floor=val.coverage_floor,
        coverage_lower_bound=lower,
        estimators=rows,
        cuped_variance_reduction=cuped_vr,
        cuped_threshold=val.cuped_min_variance_reduction,
        cuped_pass=cuped_pass,
        srm_imbalanced_detected=srm_imb.is_srm,
        srm_balanced_cleared=not srm_bal.is_srm,
        srm_pass=srm_imb.is_srm and not srm_bal.is_srm,
        config_hash=config.config_hash(),
    )


def _ok(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def format_report(report: ValidationReport) -> str:
    """Render the validation report as a clean text table."""
    lines = [
        f"LiftLab validation, {report.n_simulations} Monte-Carlo replicates, "
        f"N={report.sample_size:,} per experiment",
        f"(synthetic injected effects; thresholds from config; config_hash={report.config_hash})",
        "",
        f"{'estimator':<20}{'coverage':>11}{'gate':>7}{'A/A FPR':>11}{'gate':>7}",
        "-" * 56,
    ]
    for row in report.estimators:
        lines.append(
            f"{row.name:<20}{row.coverage:>10.1%}{_ok(row.coverage_pass):>7}"
            f"{row.fpr:>10.1%}{_ok(row.fpr_pass):>7}"
        )
    lines += [
        "-" * 56,
        f"coverage gate: observed >= {report.coverage_lower_bound:.3f}  "
        f"(nominal {report.coverage_target:.0%}; finite-sample floor "
        f"{report.coverage_floor:.0%}, MC-noise-aware)",
        f"A/A FPR gate:  |observed - {report.alpha:.0%}| <= tolerance",
        "",
        f"CUPED variance reduction: {report.cuped_variance_reduction:>6.1%}  "
        f"(>= {report.cuped_threshold:.0%})   {_ok(report.cuped_pass)}",
        f"SRM detection: imbalanced flagged={report.srm_imbalanced_detected}, "
        f"balanced cleared={report.srm_balanced_cleared}   {_ok(report.srm_pass)}",
        "",
        f"OVERALL: {_ok(report.all_passed)}",
    ]
    return "\n".join(lines)
