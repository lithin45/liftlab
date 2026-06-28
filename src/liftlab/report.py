"""Decision report ("report card") for a single experiment.

Ties the engine together for one experiment: power/MDE for the design, the point
estimate + CI per metric (naive, and CUPED for the continuous metric), the SRM verdict,
and a ship / don't-ship / inconclusive recommendation, always alongside the disclosed
true effect. Consumed by ``make demo`` (text) and the Streamlit report card.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from liftlab.config import Config, load_config
from liftlab.cuped.cuped import CupedResult, cuped_estimate
from liftlab.data.download import population_is_synthetic
from liftlab.estimators.power import (
    mde_two_means,
    mde_two_proportions,
    power_two_means,
    power_two_proportions,
)
from liftlab.estimators.twosample import EffectEstimate, estimate_effect
from liftlab.simulation.simulate import sample_population_units, simulate_experiment
from liftlab.srm.srm import SRMResult, srm_check


@dataclass
class MetricReport:
    name: str
    metric_type: str
    true_effect: float  # disclosed ground truth
    estimate: EffectEstimate  # naive
    cuped: CupedResult | None  # continuous only
    mde: float
    power_at_truth: float
    decision: str

    @property
    def primary(self) -> EffectEstimate:
        """The recommended estimate (CUPED-adjusted when available)."""
        return self.cuped.adjusted if self.cuped is not None else self.estimate


@dataclass
class DecisionReport:
    experiment_name: str
    config_hash: str
    sample_size: int
    assignment_ratio: float
    population_is_synthetic: bool | None
    srm: SRMResult
    metrics: list[MetricReport] = field(default_factory=list)


def _decide(primary: EffectEstimate, srm: SRMResult) -> str:
    if srm.is_srm:
        return "INVALID: sample ratio mismatch (results untrustworthy)"
    if primary.ci_low > 0:
        return "SHIP: CI excludes 0 (positive)"
    if primary.ci_high < 0:
        return "DO NOT SHIP: CI excludes 0 (negative)"
    return "INCONCLUSIVE: CI contains 0"


def build_decision_report(
    config: Config | None = None, db_path: Path | None = None, seed: int | None = None
) -> DecisionReport:
    """Run one experiment from the warehouse population and assemble the report card."""
    config = config or load_config()
    seed = config.seed if seed is None else seed
    alpha, power = config.power.alpha, config.power.power

    covariate = sample_population_units(config, db_path, seed=config.seed)
    result = simulate_experiment(config, covariate, seed=seed)
    units, design = result.units, result.design
    n_t, n_c = design["n_treatment"], design["n_control"]

    variant = units["variant"].to_numpy()
    z = units["covariate"].to_numpy()

    srm = srm_check(n_c, n_t, config.experiment.assignment_ratio, config.validation.srm_alpha)

    # --- Revenue (continuous): naive + CUPED ---
    rev_cfg = config.experiment.metrics.revenue
    rev_naive = estimate_effect(units["y_revenue"].to_numpy(), variant, "continuous", alpha)
    rev_cuped = cuped_estimate(units["y_revenue"].to_numpy(), z, variant, alpha)
    revenue = MetricReport(
        name=rev_cfg.name,
        metric_type="continuous",
        true_effect=design["revenue"]["effective_true_ate"],
        estimate=rev_naive,
        cuped=rev_cuped,
        mde=mde_two_means(rev_cfg.outcome_sd, n_t, n_c, alpha, power),
        power_at_truth=power_two_means(
            design["revenue"]["effective_true_ate"], rev_cfg.outcome_sd, n_t, n_c, alpha
        ),
        decision=_decide(rev_cuped.adjusted, srm),
    )

    # --- Conversion (proportion): naive ---
    conv_cfg = config.experiment.metrics.conversion
    conv_naive = estimate_effect(units["y_conversion"].to_numpy(), variant, "proportion", alpha)
    conv_truth = design["conversion"]["effective_true_ate"]
    conversion = MetricReport(
        name=conv_cfg.name,
        metric_type="proportion",
        true_effect=conv_truth,
        estimate=conv_naive,
        cuped=None,
        mde=mde_two_proportions(conv_cfg.base_rate, n_t, n_c, alpha, power),
        power_at_truth=power_two_proportions(
            conv_cfg.base_rate, conv_cfg.base_rate + conv_truth, n_t, n_c, alpha
        ),
        decision=_decide(conv_naive, srm),
    )

    return DecisionReport(
        experiment_name=config.experiment.name,
        config_hash=config.config_hash(),
        sample_size=design["sample_size"],
        assignment_ratio=config.experiment.assignment_ratio,
        population_is_synthetic=population_is_synthetic(),
        srm=srm,
        metrics=[revenue, conversion],
    )


def build_causal_demo(
    config: Config | None = None,
    db_path: Path | None = None,
    seed: int | None = None,
    *,
    confounding: float = 1.0,
    n_boot: int = 200,
    with_dowhy: bool = False,
):
    """Build the broken-randomization causal-fallback demo (confounded scenario + estimates)."""
    from liftlab.causal.estimators import run_causal_analysis
    from liftlab.causal.scenario import make_confounded_scenario

    config = config or load_config()
    seed = config.seed if seed is None else seed
    covariate = sample_population_units(config, db_path, seed=config.seed)
    scenario = make_confounded_scenario(config, covariate, seed=seed, confounding=confounding)
    result = run_causal_analysis(
        scenario.data, scenario.true_effect, n_boot=n_boot, seed=seed, with_dowhy=with_dowhy
    )
    return scenario, result


def format_causal_demo(scenario, result) -> str:
    """Render the causal fallback demo as a text block."""
    naive, did, ipw = result.naive, result.did, result.ipw
    lines = [
        "=== Causal fallback: when randomization breaks ===",
        "Confounded assignment: treatment probability depends on pre-period spend, so the",
        f"groups are not comparable and the naive estimate is biased. True effect = {result.true_effect:+.3f}.",
        "",
        f"  Naive (post-only):        {naive.estimate:+.3f}  95% CI [{naive.ci_low:+.3f}, {naive.ci_high:+.3f}]  <- BIASED",
        f"  Difference-in-Diff (DiD): {did.estimate:+.3f}  95% CI [{did.ci_low:+.3f}, {did.ci_high:+.3f}]  "
        f"covers truth={did.ci_contains(result.true_effect)}",
        f"  Inverse-propensity (IPW): {ipw.estimate:+.3f}  95% CI [{ipw.ci_low:+.3f}, {ipw.ci_high:+.3f}]  "
        f"covers truth={ipw.ci_contains(result.true_effect)}",
        f"  Placebo (mean permuted T): {result.placebo_effect:+.3f}  <- ~0 confirms validity",
    ]
    if result.dowhy_ipw is not None:
        lines.append(f"  DoWhy IPW (cross-check):   {result.dowhy_ipw:+.3f}")
        if result.refutations:
            refs = ", ".join(
                f"{k}={v:+.3f}" for k, v in result.refutations.items() if v is not None
            )
            lines.append(f"  DoWhy refuters (stable):   {refs}")
    lines += [
        "",
        "Assumptions: DiD needs parallel trends; IPW needs unconfoundedness + overlap "
        f"(propensity in [{ipw.min_propensity:.2f}, {ipw.max_propensity:.2f}]).",
    ]
    return "\n".join(lines)


def format_decision_card(report: DecisionReport) -> str:
    """Render the decision report as a text card for the CLI."""
    pop = (
        "SYNTHETIC stand-in"
        if report.population_is_synthetic
        else "REAL Olist"
        if report.population_is_synthetic is False
        else "UNKNOWN-provenance"
    )
    srm = report.srm
    lines = [
        f"=== LiftLab Decision Card: {report.experiment_name} ===",
        "DISCLOSURE: synthetic experiment. The true effects below are INJECTED by us and",
        "shown so the report can never be mistaken for a real-world result.",
        "",
        f"Population: {pop}   N={report.sample_size:,}   "
        f"assignment={report.assignment_ratio:.0%} treatment   config_hash={report.config_hash}",
        f"SRM check: {'FAIL (imbalanced!)' if srm.is_srm else 'PASS'}  "
        f"(observed ratio {srm.observed_ratio:.4f}, p={srm.p_value:.3g})",
    ]
    for m in report.metrics:
        lines += ["", f"Metric: {m.name} ({m.metric_type})"]
        lines.append(f"  True effect (disclosed):  {m.true_effect:+.4f}")
        e = m.estimate
        lines.append(
            f"  Naive estimate:           {e.estimate:+.4f}  95% CI [{e.ci_low:+.4f}, {e.ci_high:+.4f}]  p={e.p_value:.3g}"
        )
        if m.cuped is not None:
            a = m.cuped.adjusted
            lines.append(
                f"  CUPED estimate (rec.):    {a.estimate:+.4f}  95% CI [{a.ci_low:+.4f}, {a.ci_high:+.4f}]  "
                f"(variance -{m.cuped.variance_reduction:.0%})"
            )
        lines.append(f"  Power at true effect: {m.power_at_truth:.2f}   MDE(80%): {m.mde:+.4f}")
        lines.append(f"  DECISION: {m.decision}")
    return "\n".join(lines)
