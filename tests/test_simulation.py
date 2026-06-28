"""Phase 2: the synthetic experiment generator recovers the known truth.

Because the effect is injected, these tests assert the simulator's *ground-truth
contract*: the realized assignment matches config, the naive estimator lands within
a few SE of the injected effect (recoverable in principle), the covariate correlation
is calibrated (so CUPED's >=30% gate holds by construction), A/A has no effect, and
everything is reproducible under a fixed seed.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from liftlab.config import load_config
from liftlab.simulation.simulate import sample_covariate, simulate_experiment
from liftlab.simulation.store import load_run, run_id, store_run

# 4 SE ~ 99.99% interval: a robust "recoverable in principle" check (and seeded).
SE_MULT = 4.0


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def covariate() -> np.ndarray:
    """A heavy-tailed pre-period covariate, like real spend."""
    rng = np.random.default_rng(12345)
    return rng.lognormal(mean=3.5, sigma=0.8, size=40000)


def _se_diff(y: np.ndarray, t: np.ndarray) -> float:
    yt, yc = y[t == 1], y[t == 0]
    return math.sqrt(yt.var(ddof=1) / yt.size + yc.var(ddof=1) / yc.size)


def _naive_diff(df, col: str) -> float:
    t = df["variant"] == 1
    return float(df.loc[t, col].mean() - df.loc[~t, col].mean())


def test_realized_ratio_matches_config(cfg, covariate) -> None:
    res = simulate_experiment(cfg, covariate, seed=1)
    assert abs(res.design["realized_ratio"] - cfg.experiment.assignment_ratio) < 0.01
    assert res.design["n_treatment"] + res.design["n_control"] == res.design["sample_size"]


def test_revenue_effect_is_recoverable(cfg, covariate) -> None:
    res = simulate_experiment(cfg, covariate, seed=1)
    diff = _naive_diff(res.units, "y_revenue")
    se = _se_diff(res.units["y_revenue"].to_numpy(), res.units["variant"].to_numpy())
    tau = cfg.experiment.metrics.revenue.true_effect_absolute
    assert abs(diff - tau) < SE_MULT * se


def test_conversion_effect_is_recoverable(cfg, covariate) -> None:
    res = simulate_experiment(cfg, covariate, seed=2)
    diff = _naive_diff(res.units, "y_conversion")
    se = _se_diff(
        res.units["y_conversion"].to_numpy().astype(float), res.units["variant"].to_numpy()
    )
    tau_p = cfg.experiment.metrics.conversion.true_lift_absolute
    assert abs(diff - tau_p) < SE_MULT * se


def test_covariate_correlation_is_calibrated(cfg, covariate) -> None:
    """corr(covariate, revenue) ~= target -> CUPED variance reduction == corr^2 >= 30%."""
    res = simulate_experiment(cfg, covariate, seed=3)
    rho = float(np.corrcoef(res.units["covariate"], res.units["y_revenue"])[0, 1])
    assert abs(rho - cfg.experiment.covariate.target_correlation) < 0.03
    assert rho**2 >= 0.30


def test_simulation_is_deterministic(cfg, covariate) -> None:
    a = simulate_experiment(cfg, covariate, seed=7)
    b = simulate_experiment(cfg, covariate, seed=7)
    assert_frame_equal(a.units, b.units)
    # design is now a pure function of (config, seed, overrides): byte-equal, no pops.
    assert a.design == b.design
    assert "generated_at" not in a.design


def test_different_seed_changes_assignment(cfg, covariate) -> None:
    a = simulate_experiment(cfg, covariate, seed=1)
    b = simulate_experiment(cfg, covariate, seed=2)
    assert not np.array_equal(a.units["variant"].to_numpy(), b.units["variant"].to_numpy())


def test_aa_simulation_has_no_effect(cfg, covariate) -> None:
    res = simulate_experiment(cfg, covariate, seed=5, revenue_effect=0.0, conversion_lift=0.0)
    assert res.design["is_aa"] is True
    assert res.design["revenue"]["true_effect_absolute"] == 0.0
    diff = _naive_diff(res.units, "y_revenue")
    se = _se_diff(res.units["y_revenue"].to_numpy(), res.units["variant"].to_numpy())
    assert abs(diff) < SE_MULT * se


def test_srm_ratio_override(cfg, covariate) -> None:
    res = simulate_experiment(cfg, covariate, seed=1, assignment_ratio=0.7)
    assert abs(res.design["realized_ratio"] - 0.7) < 0.01


def test_sample_covariate_with_and_without_replacement() -> None:
    pop = np.arange(100.0)
    drawn = sample_covariate(pop, 50, seed=0)
    assert drawn.size == 50
    assert len(np.unique(drawn)) == 50  # without replacement
    over = sample_covariate(pop, 250, seed=0)
    assert over.size == 250  # with replacement


def test_store_and_load_roundtrip(cfg, covariate, tmp_path: Path) -> None:
    res = simulate_experiment(cfg, covariate, seed=9)
    out = store_run(res, runs_dir=tmp_path)
    assert (out / "design.json").is_file()
    assert (out / "units.csv").is_file()
    assert (out / "provenance.json").is_file()
    loaded = load_run(out)
    assert loaded.design == res.design
    # Float columns must reload bit-exact (round-trip parse), not just within rtol.
    assert np.array_equal(loaded.units["y_revenue"].to_numpy(), res.units["y_revenue"].to_numpy())
    assert np.array_equal(loaded.units["covariate"].to_numpy(), res.units["covariate"].to_numpy())


def test_run_id_distinguishes_variants_at_same_seed(cfg, covariate, tmp_path: Path) -> None:
    """SRM / A/A / partial-effect variants must NOT collide on run_id (silent overwrite)."""
    base = simulate_experiment(cfg, covariate, seed=1)
    srm = simulate_experiment(cfg, covariate, seed=1, assignment_ratio=0.7)
    aa = simulate_experiment(cfg, covariate, seed=1, revenue_effect=0.0, conversion_lift=0.0)
    partial = simulate_experiment(cfg, covariate, seed=1, revenue_effect=0.0)

    ids = {run_id(base.design), run_id(srm.design), run_id(aa.design), run_id(partial.design)}
    assert len(ids) == 4  # all distinct

    # And persisting them side by side keeps four separate, loadable runs.
    for res in (base, srm, aa, partial):
        store_run(res, runs_dir=tmp_path)
    run_dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(run_dirs) == 4
    assert load_run(run_dirs[0]).design["seed"] == 1


def test_design_generative_params_match_config(cfg, covariate) -> None:
    """Guards against a config-parse regression silently changing the generative model."""
    d = simulate_experiment(cfg, covariate, seed=4).design
    rev, conv = cfg.experiment.metrics.revenue, cfg.experiment.metrics.conversion
    rho = cfg.experiment.covariate.target_correlation
    assert d["revenue"]["baseline_mean"] == rev.baseline_mean
    assert d["revenue"]["outcome_sd"] == rev.outcome_sd
    assert d["revenue"]["beta"] == pytest.approx(rho * rev.outcome_sd)
    assert d["conversion"]["base_rate"] == conv.base_rate
    assert d["conversion"]["covariate_coef"] == conv.covariate_coef
    # Effective true ATE matches nominal (clipping negligible at configured values).
    assert d["revenue"]["effective_true_ate"] == rev.true_effect_absolute
    assert d["conversion"]["effective_true_ate"] == pytest.approx(conv.true_lift_absolute, abs=1e-4)


@pytest.mark.parametrize(
    "metric,col,truth_key",
    [
        ("revenue", "y_revenue", "true_effect_absolute"),
        ("conversion", "y_conversion", "true_lift_absolute"),
    ],
)
def test_injected_effect_is_unbiased_over_many_seeds(
    cfg, covariate, metric, col, truth_key
) -> None:
    """Average the naive estimate over many re-randomizations: it must converge to the
    injected truth (gates UNBIASEDNESS, unlike the loose single-seed 4*SE check)."""
    k = 150
    diffs = []
    for seed in range(100, 100 + k):
        df = simulate_experiment(cfg, covariate, seed=seed).units
        diffs.append(_naive_diff(df, col))
    mean_diff = float(np.mean(diffs))
    se_of_mean = float(np.std(diffs, ddof=1) / math.sqrt(k))
    truth = getattr(getattr(cfg.experiment.metrics, metric), truth_key)
    assert abs(mean_diff - truth) < 4 * se_of_mean


def test_load_covariate_values_reads_warehouse(tmp_path: Path) -> None:
    import duckdb

    from liftlab.simulation.simulate import load_covariate_values

    db = tmp_path / "wh.duckdb"
    con = duckdb.connect(str(db))  # DuckDB provides the `main` schema by default
    con.execute(
        "CREATE TABLE main.customer_metrics AS SELECT * FROM "
        "(VALUES ('c', 10.0), ('a', 20.0), ('b', 0.0)) t(customer_unique_id, pre_period_value)"
    )
    con.close()
    values = load_covariate_values(db)
    assert sorted(values.tolist()) == [0.0, 10.0, 20.0]
    # Deterministic, stable order (ORDER BY customer_unique_id): a,b,c -> 20,0,10.
    assert values.tolist() == [20.0, 0.0, 10.0]
    assert np.array_equal(values, load_covariate_values(db))


def test_load_covariate_values_missing_warehouse_errors(tmp_path: Path) -> None:
    from liftlab.simulation.simulate import load_covariate_values

    with pytest.raises(FileNotFoundError, match="make data"):
        load_covariate_values(tmp_path / "nope.duckdb")
