"""LiftLab Streamlit experiment report card.

Run via `make up` (Docker) or `uv run --group ui streamlit run src/liftlab/ui/app.py`.
Also deploy-ready for Streamlit Community Cloud: it builds its own demo data on first load.

Leads with the disclosure, then the decision cards, the CUPED variance reduction chart,
the SRM verdict, the A/A calibration plus coverage validation, and the causal fallback.
Every effect is shown against the disclosed injected truth.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the `src/` layout importable when deployed (e.g. Streamlit Community Cloud),
# where the project package may not be pip-installed.
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from liftlab.config import load_config  # noqa: E402
from liftlab.paths import duckdb_path  # noqa: E402
from liftlab.report import build_causal_demo, build_decision_report  # noqa: E402
from liftlab.simulation.validation import run_validation  # noqa: E402

st.set_page_config(page_title="LiftLab: Experiment Report Card", page_icon="🧪", layout="wide")

# --------------------------------------------------------------------------- #
# Disclosure (always first)
# --------------------------------------------------------------------------- #
st.title("🧪 LiftLab: Experiment Report Card")
st.warning(
    "**Synthetic design disclosure.** This is not a real world experiment. The treatment "
    "effect is **injected by us**, so the true effect is known. That is exactly what lets "
    "every estimator be validated against ground truth, and the true effect is shown next to "
    "each estimate below. The population is the real Olist dataset (CC BY-NC-SA 4.0, "
    "non commercial) when available, or a disclosed synthetic stand in."
)
st.caption(
    "Free, self contained demo. It uses no external APIs, accounts, or keys, so it runs "
    "entirely on open source libraries."
)

cfg = load_config()


@st.cache_resource(show_spinner="Preparing the demo data (first run only, takes a moment)...")
def _ensure_warehouse() -> bool:
    """Build the synthetic demo warehouse if it is missing (so the app self-bootstraps
    when deployed, where `make data` has not been run)."""
    import os

    if not duckdb_path().is_file():
        os.environ.setdefault("LIFTLAB_DATA_SOURCE", "synthetic")
        from liftlab.data.build import build

        build()
    return True


_ensure_warehouse()


# --------------------------------------------------------------------------- #
# Cached compute
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _report(seed: int):
    return build_decision_report(seed=seed)


@st.cache_data(show_spinner=False)
def _validation(m: int):
    return run_validation(n_simulations=m)


@st.cache_data(show_spinner=False)
def _causal(seed: int, confounding: float):
    return build_causal_demo(seed=seed, confounding=confounding, with_dowhy=True, n_boot=200)


def _interval_plot(rows, truth, title, xlabel):
    """Horizontal point + 95% CI plot with a line at the disclosed truth and at 0."""
    fig = go.Figure()
    labels = [r[0] for r in rows]
    for label, est, lo, hi in rows:
        fig.add_trace(
            go.Scatter(
                x=[est],
                y=[label],
                error_x={"type": "data", "array": [hi - est], "arrayminus": [est - lo]},
                mode="markers",
                marker={"size": 13},
                showlegend=False,
            )
        )
    fig.add_vline(x=0, line_color="#888", line_width=1)
    fig.add_vline(
        x=truth,
        line_dash="dash",
        line_color="#2ca02c",
        annotation_text=f"true {truth:+.3g}",
        annotation_position="top",
    )
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        height=80 + 60 * len(labels),
        margin={"l": 10, "r": 10, "t": 50, "b": 30},
        yaxis={"categoryorder": "array", "categoryarray": labels[::-1]},
    )
    return fig


# --------------------------------------------------------------------------- #
# Sidebar: experiment design (disclosed)
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Experiment design")
    st.caption(f"`{cfg.experiment.name}`  ·  config `{cfg.config_hash()}`")
    seed = st.number_input("Experiment seed", value=int(cfg.seed), step=1)
    st.markdown("**Disclosed true effects**")
    st.write(
        {
            "revenue ATE": cfg.experiment.metrics.revenue.true_effect_absolute,
            "conversion lift": cfg.experiment.metrics.conversion.true_lift_absolute,
            "assignment": f"{cfg.experiment.assignment_ratio:.0%} treatment",
            "sample size": cfg.experiment.sample_size,
        }
    )

report = _report(int(seed))

# --------------------------------------------------------------------------- #
# Decision cards
# --------------------------------------------------------------------------- #
st.subheader("Decision")
pop = {True: "synthetic stand-in", False: "real Olist"}.get(
    report.population_is_synthetic, "unknown-provenance"
)
st.caption(f"Population: {pop}  ·  N={report.sample_size:,}  ·  one randomized experiment draw")

cols = st.columns(len(report.metrics))
for col, m in zip(cols, report.metrics, strict=True):
    with col:
        primary = m.primary
        verdict = m.decision.split(":")[0]
        badge = {"SHIP": "success", "INCONCLUSIVE": "warning"}.get(verdict, "error")
        getattr(st, badge if badge in ("success", "warning", "error") else "info")(
            f"**{m.name}**: {m.decision}"
        )
        st.metric(
            label=f"Estimate ({'CUPED' if m.cuped else 'naive'})",
            value=f"{primary.estimate:+.4g}",
            delta=f"true {m.true_effect:+.4g}",
            delta_color="off",
        )
        st.caption(
            f"95% CI [{primary.ci_low:+.4g}, {primary.ci_high:+.4g}]  ·  p={primary.p_value:.2g}  "
            f"·  power@truth {m.power_at_truth:.2f}  ·  MDE₈₀ {m.mde:+.3g}"
        )

# --------------------------------------------------------------------------- #
# CUPED variance reduction (lead visual)
# --------------------------------------------------------------------------- #
st.subheader("CUPED variance reduction")
revenue = report.metrics[0]
if revenue.cuped is not None:
    c1, c2 = st.columns([2, 1])
    with c1:
        rows = [
            ("Naive", revenue.estimate.estimate, revenue.estimate.ci_low, revenue.estimate.ci_high),
            (
                "CUPED",
                revenue.cuped.adjusted.estimate,
                revenue.cuped.adjusted.ci_low,
                revenue.cuped.adjusted.ci_high,
            ),
        ]
        st.plotly_chart(
            _interval_plot(
                rows, revenue.true_effect, "Revenue ATE: naive vs CUPED", "effect (BRL)"
            ),
            width="stretch",
        )
    with c2:
        st.metric("Variance reduction", f"{revenue.cuped.variance_reduction:.0%}")
        st.metric("CI width shrink", f"{revenue.cuped.se_reduction:.0%}")
        st.caption(
            f"θ={revenue.cuped.theta:.3g}, ρ={revenue.cuped.correlation:.3f}. CUPED uses the "
            "pre-period covariate to cut variance, giving a tighter CI for the *same* experiment."
        )

# --------------------------------------------------------------------------- #
# SRM
# --------------------------------------------------------------------------- #
st.subheader("Sample-Ratio Mismatch (SRM)")
srm = report.srm
s1, s2, s3 = st.columns(3)
s1.metric(
    "Observed treatment ratio", f"{srm.observed_ratio:.4f}", f"intended {srm.expected_ratio:.2f}"
)
s2.metric("χ² p-value", f"{srm.p_value:.3g}")
(s3.error if srm.is_srm else s3.success)("SRM detected!" if srm.is_srm else "Assignment OK")

# --------------------------------------------------------------------------- #
# Validation: A/A calibration + coverage
# --------------------------------------------------------------------------- #
st.subheader("Validation: A/A calibration and coverage")
st.caption(
    "Monte-Carlo over many re-randomized experiments. The full gate is `make eval` "
    "(M=3000); this card uses a lighter M for responsiveness."
)
if st.button("Run the Monte-Carlo validation"):
    with st.spinner("Running Monte-Carlo validation..."):
        val = _validation(500)

    v1, v2 = st.columns(2)
    with v1:
        fig = go.Figure()
        names = [r.name for r in val.estimators]
        fig.add_bar(x=names, y=[r.coverage for r in val.estimators], name="coverage")
        fig.add_hline(
            y=val.coverage_target,
            line_dash="dash",
            line_color="#2ca02c",
            annotation_text="95% target",
        )
        fig.update_layout(
            title="CI coverage of the true effect", yaxis_range=[0.85, 1.0], height=320
        )
        st.plotly_chart(fig, width="stretch")
    with v2:
        fig = go.Figure()
        fig.add_bar(x=names, y=[r.fpr for r in val.estimators], name="A/A FPR")
        fig.add_hline(
            y=val.alpha, line_dash="dash", line_color="#d62728", annotation_text="alpha=5%"
        )
        fig.update_layout(title="A/A false-positive rate", yaxis_range=[0.0, 0.12], height=320)
        st.plotly_chart(fig, width="stretch")

    g1, g2, g3 = st.columns(3)
    g1.metric(
        "CUPED variance reduction",
        f"{val.cuped_variance_reduction:.0%}",
        f"gate >= {val.cuped_threshold:.0%}",
    )
    g2.metric("Coverage gate floor", f"{val.coverage_lower_bound:.3f}")
    (g3.success if val.all_passed else g3.error)(
        "ALL GATES PASS" if val.all_passed else "GATE FAILURE"
    )

# --------------------------------------------------------------------------- #
# Causal fallback
# --------------------------------------------------------------------------- #
st.subheader("Causal fallback: when randomization breaks")
st.caption(
    "If assignment is confounded (treatment depends on pre-period spend), the naive estimate "
    "is biased. Difference in differences and inverse propensity weighting recover the truth."
)
if st.button("Run causal fallback demo (confounded assignment)"):
    with st.spinner("Estimating DiD / IPW + DoWhy refuters..."):
        scenario, causal = _causal(int(seed), 1.0)
    rows = [
        ("Naive (biased)", causal.naive.estimate, causal.naive.ci_low, causal.naive.ci_high),
        ("DiD", causal.did.estimate, causal.did.ci_low, causal.did.ci_high),
        ("IPW", causal.ipw.estimate, causal.ipw.ci_low, causal.ipw.ci_high),
    ]
    st.plotly_chart(
        _interval_plot(
            rows, causal.true_effect, "Confounded experiment: naive vs causal", "effect"
        ),
        width="stretch",
    )
    cc1, cc2 = st.columns(2)
    cc1.metric("Naive bias", f"{causal.naive.estimate - causal.true_effect:+.3g}")
    cc1.metric(
        "Placebo (mean permuted T)",
        f"{causal.placebo_effect:+.3g}",
        "~0 = valid",
        delta_color="off",
    )
    if causal.dowhy_ipw is not None:
        cc2.metric("DoWhy IPW (cross-check)", f"{causal.dowhy_ipw:+.3g}")
        cc2.caption(
            "Refuters (stable means robust): "
            + ", ".join(f"{k} {v:+.3g}" for k, v in causal.refutations.items() if v is not None)
        )
    st.caption(
        "**Assumptions.** DiD: parallel trends (the confounder's effect is time invariant). "
        f"IPW: unconfoundedness plus overlap (propensity in "
        f"[{causal.ipw.min_propensity:.2f}, {causal.ipw.max_propensity:.2f}])."
    )

st.divider()
st.caption(
    "LiftLab · provably correct A/B testing engine · estimators validated against injected "
    "ground truth · code MIT, Olist data CC BY-NC-SA 4.0 (non-commercial)."
)
