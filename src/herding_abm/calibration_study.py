from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from .model import ASSETS, Scenario, calibration, simulate


PARAMETERS = ("price_impact_multiplier", "depth_multiplier", "liquidity_withdrawal_strength",
              "background_order_multiplier", "reaction_time_heterogeneity")


def _drawdown(values: np.ndarray) -> float:
    wealth = np.exp(np.cumsum(values))
    peaks = np.maximum.accumulate(wealth)
    return float(np.max((peaks - wealth) / peaks))


def _historical_targets(returns: pd.DataFrame, window_days: int) -> dict:
    frame = returns.loc[:, ASSETS].dropna()
    drawdowns = {asset: [] for asset in ASSETS}
    for asset in ASSETS:
        values = frame[asset].to_numpy()
        for offset in range(0, len(values) - window_days + 1, window_days):
            drawdowns[asset].append(_drawdown(values[offset:offset + window_days]))
    return {
        "volatility": frame.std().to_numpy() * np.sqrt(252),
        "correlation": frame.corr().to_numpy(),
        "autocorrelation": np.array([frame[a].autocorr(1) for a in ASSETS]),
        "skewness": frame.skew().to_numpy(),
        "kurtosis": frame.kurt().to_numpy(),
        "drawdown_median": np.array([np.median(drawdowns[a]) for a in ASSETS]),
        "drawdown_q90": np.array([np.quantile(drawdowns[a], .90) for a in ASSETS]),
        "drawdowns": drawdowns,
    }


def _simulate_targets(config: dict, calib: dict, seeds: list[int]) -> dict:
    returns_by_asset = {asset: [] for asset in ASSETS}
    drawdowns = {asset: [] for asset in ASSETS}
    for seed in seeds:
        frame, _ = simulate(config, Scenario("traditional", 0.0, 0.0, "none", "none", seed), calib)
        for asset in ASSETS:
            values = frame[f"return_{asset}"].iloc[1:].to_numpy()
            returns_by_asset[asset].append(values)
            drawdowns[asset].append(_drawdown(values))
    matrix = np.column_stack([np.concatenate(returns_by_asset[a]) for a in ASSETS])
    annualiser = np.sqrt(252 * float(config.get("intervals_per_day", 24)))
    return {
        "volatility": matrix.std(axis=0, ddof=1) * annualiser,
        "correlation": np.corrcoef(matrix.T),
        "autocorrelation": np.array([pd.Series(matrix[:, i]).autocorr(1) for i in range(len(ASSETS))]),
        "skewness": pd.DataFrame(matrix, columns=ASSETS).skew().to_numpy(),
        "kurtosis": pd.DataFrame(matrix, columns=ASSETS).kurt().to_numpy(),
        "drawdown_median": np.array([np.median(drawdowns[a]) for a in ASSETS]),
        "drawdown_q90": np.array([np.quantile(drawdowns[a], .90) for a in ASSETS]),
        "drawdowns": drawdowns,
    }


def _errors(historical: dict, simulated: dict, weights: dict) -> dict[str, float]:
    rel = lambda actual, estimate, floor=1e-8: np.abs(estimate - actual) / np.maximum(np.abs(actual), floor)
    volatility = float(rel(historical["volatility"], simulated["volatility"]).mean())
    drawdown = float((rel(historical["drawdown_median"], simulated["drawdown_median"]) +
                      rel(historical["drawdown_q90"], simulated["drawdown_q90"])).mean() / 2)
    upper = np.triu_indices(len(ASSETS), 1)
    correlation = float(np.abs(simulated["correlation"][upper] - historical["correlation"][upper]).mean())
    autocorrelation = float(np.abs(simulated["autocorrelation"] - historical["autocorrelation"]).mean())
    skew_error = rel(historical["skewness"], simulated["skewness"], .25).mean()
    kurt_error = rel(historical["kurtosis"], simulated["kurtosis"], 1.0).mean()
    tails = float((skew_error + kurt_error) / 2)
    loss = (weights["volatility"] * volatility + weights["drawdown"] * drawdown +
            weights["correlation"] * correlation + weights["autocorrelation"] * autocorrelation +
            weights["tails"] * tails)
    return {"volatility_error": volatility, "drawdown_error": drawdown,
            "correlation_error": correlation, "autocorrelation_error": autocorrelation,
            "tail_error": tails, "weighted_loss": float(loss)}


def _quantile_rows(label: str, targets: dict) -> list[dict]:
    rows = []
    for index, asset in enumerate(ASSETS):
        values = targets["drawdowns"][asset]
        rows.append({"Dataset": label, "Asset": asset, "Count": len(values),
                     "Median": float(np.median(values)), "Q75": float(np.quantile(values, .75)),
                     "Q90": float(np.quantile(values, .90)), "Q95": float(np.quantile(values, .95)),
                     "Maximum": float(np.max(values))})
    return rows


def run_calibration_study(spec: dict, output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = json.loads(Path(spec["base_config"]).read_text(encoding="utf-8"))
    # The split-sample study was introduced after reaction-time heterogeneity.
    # Record its stream convention explicitly so its archived 0.6% holdout
    # comparison remains reproducible without changing the primary v1 grid.
    base["random_stream_version"] = spec.get("random_stream_version", "calibration_v2")
    prices_csv = base["data"]["prices_csv"]
    returns = pd.read_csv(Path(prices_csv).parent / "etf_log_returns.csv", parse_dates=["Date"]).set_index("Date")
    calibration_returns = returns.loc[spec["calibration_start"]:spec["calibration_end"]]
    holdout_returns = returns.loc[spec["holdout_start"]:spec["holdout_end"]]
    window_days = max(5, round(base["steps"] / float(base.get("intervals_per_day", 24))))
    historical_calibration = _historical_targets(calibration_returns, window_days)
    historical_holdout = _historical_targets(holdout_returns, window_days)
    calibration_parameters = calibration(prices_csv, start=spec["calibration_start"], end=spec["calibration_end"])

    candidate_rows, error_rows = [], []
    baseline_parameters = {"price_impact_multiplier": 1.0, "depth_multiplier": 1.0,
                           "liquidity_withdrawal_strength": 1.0, "background_order_multiplier": 1.0,
                           "reaction_time_heterogeneity": 0.0}
    baseline_simulated_calibration = _simulate_targets(base, calibration_parameters, spec["seeds"])
    candidate_rows.append({"candidate_id": 0, **baseline_parameters})
    error_rows.append({"candidate_id": 0, **baseline_parameters,
                       **_errors(historical_calibration, baseline_simulated_calibration, spec["loss_weights"])})
    combinations = product(*(spec[name] for name in PARAMETERS))
    for candidate_id, values in enumerate(combinations, 1):
        parameters = dict(zip(PARAMETERS, values))
        candidate_rows.append({"candidate_id": candidate_id, **parameters})
        candidate_config = dict(base); candidate_config.update(parameters)
        simulated = _simulate_targets(candidate_config, calibration_parameters, spec["seeds"])
        error_rows.append({"candidate_id": candidate_id, **parameters,
                           **_errors(historical_calibration, simulated, spec["loss_weights"])})
    candidates = pd.DataFrame(candidate_rows)
    errors = pd.DataFrame(error_rows).sort_values("weighted_loss").reset_index(drop=True)
    selected = errors.iloc[0]
    selected_parameters = {name: float(selected[name]) for name in PARAMETERS}
    selected_payload = {
        "calibration_period": [spec["calibration_start"], spec["calibration_end"]],
        "holdout_period": [spec["holdout_start"], spec["holdout_end"]],
        "window_days": window_days, "loss_weights": spec["loss_weights"],
        "selected_candidate_id": int(selected["candidate_id"]),
        "selected_parameters": selected_parameters,
        "calibration_weighted_loss": float(selected["weighted_loss"]),
    }
    selected_config = dict(base); selected_config.update(selected_parameters)
    holdout_seeds = spec.get("holdout_seeds") or [1009 + 37 * index for index in range(int(spec.get("holdout_seed_count", 50)))]
    baseline_holdout = _simulate_targets(base, calibration_parameters, holdout_seeds)
    selected_holdout = _simulate_targets(selected_config, calibration_parameters, holdout_seeds)
    holdout_rows = []
    for label, targets in (("baseline", baseline_holdout), ("selected", selected_holdout)):
        holdout_rows.append({"model": label, **_errors(historical_holdout, targets, spec["loss_weights"])})
    baseline_holdout_loss = float(holdout_rows[0]["weighted_loss"])
    selected_holdout_loss = float(holdout_rows[1]["weighted_loss"])
    holdout_improvement_pct = 100 * (baseline_holdout_loss - selected_holdout_loss) / baseline_holdout_loss
    materially_better = holdout_improvement_pct >= 5.0
    selected_payload.update({
        "holdout_seed_count": len(holdout_seeds),
        "baseline_holdout_weighted_loss": baseline_holdout_loss,
        "selected_holdout_weighted_loss": selected_holdout_loss,
        "holdout_improvement_pct": holdout_improvement_pct,
        "material_improvement_threshold_pct": 5.0,
        "adoption_decision": "adopt minimum-loss candidate" if materially_better else "retain baseline; use minimum-loss candidate as sensitivity specification",
        "decision_reason": "Out-of-sample weighted-loss improvement must be at least 5% to justify replacing the more parsimonious baseline."
    })
    quantiles = (_quantile_rows("historical_calibration", historical_calibration) +
                 _quantile_rows("historical_holdout", historical_holdout) +
                 _quantile_rows("simulated_baseline_holdout", baseline_holdout) +
                 _quantile_rows("simulated_selected_holdout", selected_holdout))

    paths = {
        "candidates": output_dir / "candidate_parameters.csv",
        "errors": output_dir / "calibration_errors.csv",
        "holdout": output_dir / "holdout_validation.csv",
        "quantiles": output_dir / "drawdown_quantiles.csv",
        "selected": output_dir / "selected_parameters.json",
        "decision": output_dir / "calibration_decision.json",
    }
    candidates.to_csv(paths["candidates"], index=False)
    errors.to_csv(paths["errors"], index=False)
    pd.DataFrame(holdout_rows).to_csv(paths["holdout"], index=False)
    pd.DataFrame(quantiles).to_csv(paths["quantiles"], index=False)
    paths["selected"].write_text(json.dumps(selected_payload, indent=2), encoding="utf-8")
    paths["decision"].write_text(json.dumps({key: selected_payload[key] for key in (
        "baseline_holdout_weighted_loss", "selected_holdout_weighted_loss", "holdout_improvement_pct",
        "material_improvement_threshold_pct", "adoption_decision", "decision_reason")}, indent=2), encoding="utf-8")
    return paths
