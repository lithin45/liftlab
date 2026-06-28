"""LiftLab command-line entry point.

Subcommands map to the Makefile targets. Phases 2/5/6 fill in ``simulate``,
``eval``, and ``demo``; in earlier phases they are clearly-labelled no-ops so the
skeleton (and ``docker compose up``) stays green.
"""

from __future__ import annotations

import argparse
import sys

from liftlab.config import load_config


def _cmd_build(args: argparse.Namespace) -> int:
    from liftlab.data.build import build

    build(force_data=args.force_data)
    return 0


def _cmd_data(args: argparse.Namespace) -> int:
    from liftlab.data.download import ensure_raw_data

    cfg = load_config()
    manifest = ensure_raw_data(cfg, force=args.force_data)
    print(
        f"[liftlab] data ready: {manifest['source_used']} "
        f"(synthetic={manifest['is_synthetic_population']})"
    )
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    from liftlab.simulation.simulate import sample_population_units, simulate_experiment
    from liftlab.simulation.store import store_run

    cfg = load_config()
    covariate = sample_population_units(cfg)
    result = simulate_experiment(cfg, covariate, seed=cfg.seed)
    out = store_run(result)

    d = result.design
    df = result.units
    treat, control = df["variant"] == 1, df["variant"] == 0
    rev_naive = df.loc[treat, "y_revenue"].mean() - df.loc[control, "y_revenue"].mean()
    conv_naive = df.loc[treat, "y_conversion"].mean() - df.loc[control, "y_conversion"].mean()

    print(f"[liftlab] simulated experiment '{d['name']}' -> {out}")
    print(
        f"[liftlab]   N={d['sample_size']:,}  realized ratio={d['realized_ratio']:.4f} "
        f"(intended {d['assignment_ratio_intended']:.2f})"
    )
    print("[liftlab]   DISCLOSED ground truth vs. naive diff-in-means (sanity preview):")
    print(
        f"[liftlab]     revenue:    true ATE={d['revenue']['true_effect_absolute']:+.3f}  "
        f"naive={rev_naive:+.3f}"
    )
    print(
        f"[liftlab]     conversion: true ATE={d['conversion']['true_lift_absolute']:+.4f}  "
        f"naive={conv_naive:+.4f}"
    )
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from liftlab.simulation.validation import format_report, run_validation

    report = run_validation()
    print(format_report(report))
    return 0 if report.all_passed else 1


def _cmd_demo(args: argparse.Namespace) -> int:
    from liftlab.data.build import build
    from liftlab.report import (
        build_causal_demo,
        build_decision_report,
        format_causal_demo,
        format_decision_card,
    )

    build(force_data=args.force_data)
    print()
    print(format_decision_card(build_decision_report()))
    print()
    scenario, causal = build_causal_demo()  # main-deps path (DoWhy cross-check is UI/test-only)
    print(format_causal_demo(scenario, causal))
    print("\n[liftlab] Run `make eval` for the validation gates, or `make up` for the report card.")
    return 0


def _cmd_pipeline(args: argparse.Namespace) -> int:
    """Full pipeline used by `docker compose up` (build -> simulate -> eval)."""
    from liftlab.data.build import build

    build(force_data=args.force_data)
    _cmd_simulate(args)
    _cmd_eval(args)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="liftlab", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, fn, help_: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_)
        p.add_argument(
            "--force-data",
            action="store_true",
            help="Regenerate/re-download the raw population even if it exists.",
        )
        p.set_defaults(func=fn)
        return p

    add("build", _cmd_build, "Build the data layer (population -> DuckDB -> dbt).")
    add("data", _cmd_data, "Ensure the raw population data exists.")
    add("simulate", _cmd_simulate, "Run the synthetic experiment simulation.")
    add("eval", _cmd_eval, "Run the Monte-Carlo validation gates.")
    add("demo", _cmd_demo, "Build + print a decision card.")
    add("pipeline", _cmd_pipeline, "Full pipeline (build -> simulate -> eval).")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
