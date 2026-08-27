"""Rebuild the dissertation analysis from the frozen Yahoo Finance inputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*arguments: str) -> None:
    command = [sys.executable, "-m", "herding_abm.cli", *map(str, arguments)]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def write_config(source: Path, destination: Path, prices_csv: Path) -> Path:
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["data"]["prices_csv"] = str(prices_csv.resolve())
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def write_spec(source: Path, destination: Path, base_config: Path) -> Path:
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["base_config"] = str(base_config.resolve())
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--output", type=Path, default=ROOT / "reproduced")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output directory is not empty: {output}")
    data_dir = output / "data"
    config_dir = output / "configs"
    result_dir = output / "outputs"
    for directory in (data_dir, config_dir, result_dir):
        directory.mkdir(parents=True, exist_ok=True)

    prepare_args = ["prepare-data", "--raw", str(ROOT / "data" / "raw"),
                    "--output", str(data_dir)]
    if args.no_plots:
        prepare_args.append("--no-plots")
    run(*prepare_args)

    prices = data_dir / "etf_adjusted_close_2010_2025.csv"
    chosen = "quick.json" if args.mode == "quick" else "default.json"
    primary_config = write_config(ROOT / "configs" / chosen, config_dir / chosen, prices)
    primary_output = result_dir / ("quick" if args.mode == "quick" else "primary")
    run("experiment", "--config", str(primary_config), "--output", str(primary_output))

    if args.mode == "quick":
        rows = sum(1 for _ in (primary_output / "experiment_runs.csv").open(encoding="utf-8")) - 1
        if rows != 96:
            raise SystemExit(f"Quick check expected 96 runs; found {rows}")
        print(f"Quick reproduction passed: {rows} runs in {primary_output}")
        return

    run("analyse", "--runs", str(primary_output / "experiment_runs.csv"),
        "--output", str(primary_output / "analysis"))
    run("validate", "--config", str(primary_config), "--output", str(result_dir / "validation"))

    robustness_spec = write_spec(ROOT / "configs" / "robustness.json",
                                 config_dir / "robustness.json", primary_config)
    run("robustness", "--spec", str(robustness_spec), "--output", str(result_dir / "robustness"))

    calibration_spec = write_spec(ROOT / "configs" / "calibration_study.json",
                                  config_dir / "calibration_study.json", primary_config)
    run("calibrate-drawdowns", "--spec", str(calibration_spec),
        "--output", str(result_dir / "drawdown_calibration"))

    calibrated_config = write_config(ROOT / "configs" / "calibrated.json",
                                     config_dir / "calibrated.json", prices)
    calibrated_output = result_dir / "calibrated_sensitivity"
    run("experiment", "--config", str(calibrated_config), "--output", str(calibrated_output))
    run("analyse", "--runs", str(calibrated_output / "experiment_runs.csv"),
        "--output", str(calibrated_output / "analysis"))
    run("compare-experiments", "--original", str(primary_output / "experiment_runs.csv"),
        "--calibrated", str(calibrated_output / "experiment_runs.csv"),
        "--output", str(result_dir / "comparison"))

    from herding_abm.reproducibility import verify_primary_results

    report = verify_primary_results(primary_output / "experiment_runs.csv")
    errors = list(report["errors"])
    sensitivity_rows = sum(1 for _ in (calibrated_output / "experiment_runs.csv").open(encoding="utf-8")) - 1
    robustness_rows = sum(1 for _ in (result_dir / "robustness" / "robustness_runs.csv").open(encoding="utf-8")) - 1
    decision = json.loads((result_dir / "drawdown_calibration" / "calibration_decision.json").read_text(encoding="utf-8"))
    if sensitivity_rows != 4320:
        errors.append(f"Expected 4,320 sensitivity runs; found {sensitivity_rows}")
    if robustness_rows != 390:
        errors.append(f"Expected 390 robustness runs; found {robustness_rows}")
    if not abs(float(decision["holdout_improvement_pct"]) - 0.6) < 0.001:
        errors.append("Holdout improvement does not reproduce the dissertation's 0.6% result")
    if not str(decision["adoption_decision"]).startswith("retain baseline"):
        errors.append("Calibration decision does not retain the dissertation baseline")
    report.update({"ok": not errors, "errors": errors, "sensitivity_rows": sensitivity_rows,
                   "robustness_rows": robustness_rows, "calibration_decision": decision})
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["ok"]:
        raise SystemExit("Full reproduction failed verification; see verification.json")
    print(f"Full dissertation reproduction passed: {primary_output}")


if __name__ == "__main__":
    main()
