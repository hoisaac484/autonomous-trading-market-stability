from __future__ import annotations

import argparse
import json
import os
from itertools import product
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / "tmp" / "matplotlib"))

import pandas as pd

from .model import ASSETS, Scenario, calibration, simulate
from .data import download_etf_data
from .data import rebuild_from_raw
from .analysis import analyse_experiment, compare_experiments, run_robustness, validate_baseline
from .calibration_study import run_calibration_study
from .reproducibility import verify_manifest, verify_primary_results


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def plot_run(frame: pd.DataFrame, output: Path, shock_start: int, shock_end: int) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for asset in ASSETS:
        axes[0].plot(frame["step"], frame[f"price_{asset}"] / frame[f"price_{asset}"].iloc[0], label=asset)
    axes[0].set_ylabel("Normalised price")
    axes[0].legend(ncol=5, fontsize=8)
    axes[1].plot(frame["step"], frame["herding"], color="#9c2f2f")
    axes[1].set_ylabel("Herding index")
    axes[2].plot(frame["step"], frame[[f"depth_{a}" for a in ASSETS]].mean(axis=1), color="#275d80")
    axes[2].set_ylabel("Mean liquidity depth")
    axes[2].set_xlabel("Simulation interval")
    for axis in axes:
        axis.axvspan(shock_start, shock_end, color="#d99b2b", alpha=.18)
        axis.grid(alpha=.2)
    fig.suptitle("Algorithmic-herding ABM diagnostic")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def run_one(config: dict, output: Path, args: argparse.Namespace) -> None:
    output.mkdir(parents=True, exist_ok=True)
    scenario = Scenario(args.market, args.participation, args.similarity, args.shock, args.safeguard, args.seed)
    calib = calibration(config.get("data", {}).get("prices_csv"), args.seed)
    frame, metrics = simulate(config, scenario, calib)
    frame.to_csv(output / "timeseries.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output / "summary.csv", index=False)
    plot_run(frame, output / "diagnostic.png", config["shock_start"], config["shock_start"] + config["shock_duration"])
    (output / "run_config.json").write_text(json.dumps({"config": config, "scenario": scenario.__dict__}, indent=2), encoding="utf-8")
    print(pd.Series(metrics).to_string())


def run_experiment(config: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    calib = calibration(config.get("data", {}).get("prices_csv"))
    rows = []
    expected_runs = (len(config["markets"]) * len(config["autonomous_participation"]) *
                     len(config["similarity"]) * len(config["shocks"]) *
                     len(config["safeguards"]) * len(config["seeds"]))
    grid = product(config["markets"], config["autonomous_participation"], config["similarity"],
                   config["shocks"], config["safeguards"], config["seeds"])
    for market, participation, similarity, shock, safeguard, seed in grid:
        _, metrics = simulate(config, Scenario(market, participation, similarity, shock, safeguard, seed), calib)
        rows.append(metrics)
    raw = pd.DataFrame(rows)
    if len(raw) != expected_runs:
        raise RuntimeError(f"Expected {expected_runs} runs but produced {len(raw)}")
    raw.to_csv(output / "experiment_runs.csv", index=False)
    keys = ["market", "participation", "similarity", "shock", "safeguard"]
    values = [column for column in raw.columns if column not in keys + ["seed"]]
    summary = raw.groupby(keys)[values].agg(["mean", "std", "count"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    for name in values:
        summary[f"{name}_ci95"] = 1.96 * summary[f"{name}_std"] / summary[f"{name}_count"].pow(.5)
    summary.reset_index().to_csv(output / "experiment_summary.csv", index=False)
    (output / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Completed {len(raw)} runs; outputs written to {output}")


def main() -> None:
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(description="Comparative algorithmic-herding ABM")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    experiment = sub.add_parser("experiment")
    for command in (run, experiment):
        command.add_argument("--config", default="configs/quick.json")
        command.add_argument("--output", required=True)
    run.add_argument("--market", choices=["traditional", "amm"], default="traditional")
    run.add_argument("--participation", type=float, default=.5)
    run.add_argument("--similarity", type=float, default=.8)
    run.add_argument("--shock", choices=["fundamental", "volatility", "liquidity"], default="fundamental")
    run.add_argument("--safeguard", choices=["none", "execution_delay", "position_limit", "liquidity_buffer", "circuit_breaker", "all"], default="none")
    run.add_argument("--seed", type=int, default=11)
    download = sub.add_parser("download-data", help="Download and clean Yahoo Finance ETF data")
    download.add_argument("--output", default="data/etf_adjusted_close_2010_2025.csv")
    download.add_argument("--start", default="2010-01-04")
    download.add_argument("--end", default="2025-12-31")
    prepare = sub.add_parser("prepare-data", help="Rebuild Section 3.3-3.5 artifacts from raw Yahoo files")
    prepare.add_argument("--raw", default="data/raw")
    prepare.add_argument("--output", default="data")
    prepare.add_argument("--start", default="2010-01-04")
    prepare.add_argument("--end", default="2025-12-31")
    prepare.add_argument("--no-plots", action="store_true", help="Skip optional historical PNG figures")
    validate = sub.add_parser("validate", help="Validate baseline simulation against historical targets")
    validate.add_argument("--config", default="configs/default.json")
    validate.add_argument("--output", default="outputs/validation")
    analyse = sub.add_parser("analyse", help="Analyse a completed factorial experiment")
    analyse.add_argument("--runs", required=True)
    analyse.add_argument("--output", required=True)
    robustness = sub.add_parser("robustness", help="Run draft Section 4.8 robustness cases")
    robustness.add_argument("--spec", default="configs/robustness.json")
    robustness.add_argument("--output", default="outputs/robustness")
    calibration_study = sub.add_parser("calibrate-drawdowns", help="Run split-sample drawdown calibration")
    calibration_study.add_argument("--spec", default="configs/calibration_study.json")
    calibration_study.add_argument("--output", default="outputs/drawdown_calibration")
    compare = sub.add_parser("compare-experiments", help="Compare original and calibrated scenario rankings")
    compare.add_argument("--original", required=True)
    compare.add_argument("--calibrated", required=True)
    compare.add_argument("--output", required=True)
    verify = sub.add_parser("verify", help="Verify frozen inputs and dissertation headline results")
    verify.add_argument("--runs", default="outputs/historical_full_methodology/experiment_runs.csv")
    verify.add_argument("--manifest", default="reproducibility/manifest.json")
    args = parser.parse_args()
    if args.command == "download-data":
        csv_path, report_path = download_etf_data(args.output, args.start, args.end)
        print(f"Clean dataset: {csv_path}\nCleaning report: {report_path}")
        return
    if args.command == "prepare-data":
        artifacts = rebuild_from_raw(args.raw, args.output, args.start, args.end,
                                     make_plots=not args.no_plots)
        print(f"Created {len(artifacts)} historical data artifacts in {args.output}")
        return
    if args.command == "analyse":
        artifacts = analyse_experiment(args.runs, args.output)
        print(f"Created {len(artifacts)} analysis artifacts in {args.output}")
        return
    if args.command == "validate":
        paths = validate_baseline(load_config(args.config), args.output)
        print(f"Validation outputs: {paths[0]} and {paths[1]}")
        return
    if args.command == "robustness":
        path = run_robustness(load_config(args.spec), args.output)
        print(f"Robustness runs: {path}")
        return
    if args.command == "calibrate-drawdowns":
        paths = run_calibration_study(load_config(args.spec), args.output)
        print(f"Calibration study created {len(paths)} outputs in {args.output}")
        return
    if args.command == "compare-experiments":
        paths = compare_experiments(args.original, args.calibrated, args.output)
        print(f"Experiment comparison created {len(paths)} outputs in {args.output}")
        return
    if args.command == "verify":
        errors = verify_manifest(Path.cwd(), args.manifest)
        report = verify_primary_results(args.runs)
        errors.extend(report["errors"])
        print(json.dumps({"ok": not errors, "errors": errors, "headline": report["headline"]}, indent=2))
        if errors:
            raise SystemExit(1)
        return
    config, output = load_config(args.config), Path(args.output)
    if args.command == "run":
        run_one(config, output, args)
    else:
        run_experiment(config, output)


if __name__ == "__main__":
    main()
