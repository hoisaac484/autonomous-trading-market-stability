from __future__ import annotations

import json
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from .model import ASSETS, Scenario, calibration, simulate


def _moments(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    centred = values - values.mean()
    std = values.std(ddof=1)
    return float(np.mean(centred ** 3) / max(std ** 3, 1e-12)), float(np.mean(centred ** 4) / max(std ** 4, 1e-12) - 3)


def validate_baseline(config: dict, output_dir: str | Path, seeds: list[int] | None = None) -> tuple[Path, Path]:
    """Compare no-autonomous baseline paths with explicit historical targets and tolerances."""
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    seeds = seeds or [11, 29, 47, 71, 101, 131, 173, 211, 257, 307]
    prices_csv = config["data"]["prices_csv"]
    historical_returns = pd.read_csv(Path(prices_csv).parent / "etf_log_returns.csv", parse_dates=["Date"]).set_index("Date")
    historical_desc = pd.read_csv(Path(prices_csv).parent / "historical_descriptive_statistics.csv").set_index("Ticker")
    historical_corr = historical_returns[list(ASSETS)].corr().to_numpy()
    sim_config = dict(config)
    calib = calibration(prices_csv)
    simulated = {asset: [] for asset in ASSETS}
    for seed in seeds:
        frame, _ = simulate(sim_config, Scenario(market="traditional", participation=0, similarity=0,
                                                  shock="none", safeguard="none", seed=seed), calib)
        for asset in ASSETS:
            simulated[asset].append(frame[f"return_{asset}"].iloc[1:].to_numpy())
    sim_matrix = np.vstack([np.concatenate(simulated[a]) for a in ASSETS]).T
    sim_corr = np.corrcoef(sim_matrix.T)
    rows = []
    tolerances = {"AnnualisedVolatility": .40, "Skewness": 1.0, "ExcessKurtosis": 1.0,
                  "Lag1Autocorrelation": .10, "MaximumDrawdown": .60}
    intervals_per_day = float(config.get("intervals_per_day", 24))
    comparable_days = max(5, int(config["steps"] / intervals_per_day))
    for index, asset in enumerate(ASSETS):
        values = sim_matrix[:, index]
        skew, kurt = _moments(values)
        wealth = np.exp(np.cumsum(values)); dd = np.max((np.maximum.accumulate(wealth) - wealth) / np.maximum.accumulate(wealth))
        historical_asset_returns = historical_returns[asset].dropna().to_numpy()
        historical_window_drawdowns = []
        for offset in range(0, len(historical_asset_returns) - comparable_days + 1, comparable_days):
            block_wealth = np.exp(np.cumsum(historical_asset_returns[offset:offset + comparable_days]))
            historical_window_drawdowns.append(np.max((np.maximum.accumulate(block_wealth) - block_wealth) / np.maximum.accumulate(block_wealth)))
        comparable_historical_drawdown = float(np.mean(historical_window_drawdowns))
        sim_values = {"AnnualisedVolatility": values.std(ddof=1) * np.sqrt(252 * intervals_per_day), "Skewness": skew,
                      "ExcessKurtosis": kurt, "Lag1Autocorrelation": pd.Series(values).autocorr(1), "MaximumDrawdown": dd}
        for metric, simulated_value in sim_values.items():
            historical_value = comparable_historical_drawdown if metric == "MaximumDrawdown" else float(historical_desc.loc[asset, metric])
            error = simulated_value - historical_value
            relative = abs(error) if metric in {"Skewness", "Lag1Autocorrelation"} else abs(error) / max(abs(historical_value), 1e-8)
            tolerance = tolerances[metric]
            rows.append({"Asset": asset, "Metric": metric, "Historical": historical_value,
                         "Simulated": simulated_value, "AbsoluteError": abs(error), "RelativeError": relative,
                         "Tolerance": tolerance, "Pass": relative <= tolerance})
    corr_error = np.abs(sim_corr[np.triu_indices(5, 1)] - historical_corr[np.triu_indices(5, 1)])
    for pair, error in zip([f"{a}-{b}" for i, a in enumerate(ASSETS) for b in ASSETS[i+1:]], corr_error):
        rows.append({"Asset": pair, "Metric": "Correlation", "Historical": np.nan, "Simulated": np.nan,
                     "AbsoluteError": error, "RelativeError": error, "Tolerance": .15, "Pass": error <= .15})
    validation = pd.DataFrame(rows)
    path = output_dir / "baseline_validation.csv"; validation.to_csv(path, index=False)
    summary = {"tests": len(validation), "passed": int(validation["Pass"].sum()),
               "pass_rate": float(validation["Pass"].mean()), "seeds": seeds,
               "tolerance_note": "Predeclared pragmatic relative-error thresholds; correlations use absolute-point error."}
    summary_path = output_dir / "baseline_validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path, summary_path


def _ols(frame: pd.DataFrame, outcome: str) -> pd.DataFrame:
    market = (frame["market"] == "amm").astype(float)
    shock_dummies = pd.get_dummies(frame["shock"], prefix="shock", drop_first=True, dtype=float)
    safeguard_dummies = pd.get_dummies(frame["safeguard"], prefix="safeguard", drop_first=True, dtype=float)
    design = pd.concat([pd.Series(1.0, index=frame.index, name="Intercept"), frame[["participation", "similarity"]],
                        (frame["participation"] * frame["similarity"]).rename("participation_x_similarity"),
                        market.rename("amm"), (frame["participation"] * market).rename("participation_x_amm"),
                        shock_dummies, safeguard_dummies], axis=1).astype(float)
    y, x = frame[outcome].to_numpy(float), design.to_numpy(float)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta; dof = max(len(y) - x.shape[1], 1)
    covariance = (residual @ residual / dof) * np.linalg.pinv(x.T @ x)
    se = np.sqrt(np.maximum(np.diag(covariance), 0)); z = beta / np.maximum(se, 1e-12)
    p = np.array([2 * (1 - .5 * (1 + erf(abs(value) / sqrt(2)))) for value in z])
    sd_y = np.std(y, ddof=1)
    standardised = beta * np.array([1 if name == "Intercept" else design[name].std(ddof=1) for name in design]) / max(sd_y, 1e-12)
    return pd.DataFrame({"Outcome": outcome, "Term": design.columns, "Coefficient": beta, "StdError": se,
                         "NormalApproxP": p, "CI95Low": beta - 1.96 * se, "CI95High": beta + 1.96 * se,
                         "StandardisedEffect": standardised})


def analyse_experiment(runs_csv: str | Path, output_dir: str | Path) -> dict[str, Path]:
    """Produce period contrasts, regressions, effects, and transparent H1-H4 evidence tables."""
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(runs_csv)
    outcomes = ["realised_volatility", "maximum_drawdown", "mean_pairwise_correlation",
                "normalised_liquidity_deterioration", "recovery_time", "peak_herding", "peak_asset_herding"]
    regressions = pd.concat([_ols(runs, outcome) for outcome in outcomes], ignore_index=True)
    regressions.to_csv(output_dir / "factorial_regressions.csv", index=False)
    contrasts = runs.assign(
        shock_minus_pre_volatility=runs["shock_volatility"] - runs["pre_volatility"],
        recovery_minus_pre_volatility=runs["recovery_volatility"] - runs["pre_volatility"],
        shock_minus_pre_correlation=runs["shock_correlation"] - runs["pre_correlation"],
        recovery_minus_pre_correlation=runs["recovery_correlation"] - runs["pre_correlation"],
        shock_liquidity_deterioration=1 - runs["shock_depth"] / runs["pre_depth"],
        recovery_liquidity_deterioration=1 - runs["recovery_depth"] / runs["pre_depth"],
    )
    contrast_columns = [c for c in contrasts if "minus_pre" in c or "liquidity_deterioration" in c]
    contrast_summary = contrasts.groupby(["market", "participation", "similarity", "shock", "safeguard"])[contrast_columns].agg(["mean", "std", "count"])
    contrast_summary.columns = [f"{a}_{b}" for a, b in contrast_summary.columns]
    contrast_summary.reset_index().to_csv(output_dir / "period_contrasts.csv", index=False)
    effects = runs.groupby(["market", "shock", "safeguard", "participation", "similarity"])[outcomes].mean().reset_index()
    effects.to_csv(output_dir / "factorial_effect_means.csv", index=False)
    hypotheses = pd.DataFrame([
        {"Hypothesis": "H1", "EvidenceTerms": "participation; participation_x_amm", "PrimaryOutcomes": "realised_volatility; mean_pairwise_correlation"},
        {"Hypothesis": "H2", "EvidenceTerms": "similarity; participation_x_similarity", "PrimaryOutcomes": "peak_herding; peak_asset_herding; realised_volatility"},
        {"Hypothesis": "H3", "EvidenceTerms": "amm; participation_x_amm", "PrimaryOutcomes": "normalised_liquidity_deterioration; realised_volatility"},
        {"Hypothesis": "H4", "EvidenceTerms": "safeguard_*", "PrimaryOutcomes": "realised_volatility; maximum_drawdown; recovery_time"},
    ])
    hypotheses.to_csv(output_dir / "hypothesis_test_map.csv", index=False)
    return {name: output_dir / name for name in ("factorial_regressions.csv", "period_contrasts.csv", "factorial_effect_means.csv", "hypothesis_test_map.csv")}


def run_robustness(spec: dict, output_dir: str | Path) -> Path:
    """Run the parameter variations explicitly listed in draft Section 4.8."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = json.loads(Path(spec["base_config"]).read_text(encoding="utf-8"))
    calib = calibration(base["data"]["prices_csv"])
    rows = []
    for case in spec["cases"]:
        case_config = dict(base)
        case_config.update({key: value for key, value in case.items() if key != "name"})
        for market in spec["markets"]:
            for shock in spec["shocks"]:
                for seed in spec["seeds"]:
                    _, metrics = simulate(case_config, Scenario(market, spec["participation"], spec["similarity"], shock, "none", seed), calib)
                    metrics["robustness_case"] = case["name"]
                    rows.append(metrics)
    raw = pd.DataFrame(rows)
    path = output_dir / "robustness_runs.csv"
    raw.to_csv(path, index=False)
    values = ["realised_volatility", "maximum_drawdown", "mean_pairwise_correlation",
              "normalised_liquidity_deterioration", "recovery_time"]
    summary = raw.groupby(["robustness_case", "market", "shock"])[values].agg(["mean", "std", "count"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(output_dir / "robustness_summary.csv", index=False)
    return path


def compare_experiments(original_csv: str | Path, calibrated_csv: str | Path,
                        output_dir: str | Path) -> dict[str, Path]:
    """Assess whether principal rankings and hypothesis directions survive recalibration."""
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    original, calibrated = pd.read_csv(original_csv), pd.read_csv(calibrated_csv)
    keys = ["market", "participation", "similarity", "shock", "safeguard"]
    outcomes = ["realised_volatility", "maximum_drawdown", "mean_pairwise_correlation",
                "normalised_liquidity_deterioration", "recovery_time", "peak_herding", "peak_asset_herding"]
    left = original.groupby(keys)[outcomes].mean().reset_index()
    right = calibrated.groupby(keys)[outcomes].mean().reset_index()
    merged = left.merge(right, on=keys, suffixes=("_original", "_calibrated"))
    ranking_rows = []
    for outcome in outcomes:
        a, b = merged[f"{outcome}_original"], merged[f"{outcome}_calibrated"]
        ranking_rows.append({"Outcome": outcome, "ScenarioRankCorrelation": a.rank().corr(b.rank()),
                             "OriginalMean": a.mean(), "CalibratedMean": b.mean(),
                             "MeanMagnitudeChangePct": (b.mean() / a.mean() - 1) * 100 if abs(a.mean()) > 1e-12 else np.nan})
    ranking = pd.DataFrame(ranking_rows)
    ranking_path = output_dir / "ranking_preservation.csv"; ranking.to_csv(ranking_path, index=False)

    def evidence(frame: pd.DataFrame, label: str) -> list[dict]:
        rows = []
        trad = frame[frame.market == "traditional"]
        low, high = trad[trad.participation == 0], trad[trad.participation == .75]
        for outcome in ("realised_volatility", "maximum_drawdown", "mean_pairwise_correlation"):
            rows.append({"Model": label, "Hypothesis": "H1", "Measure": outcome,
                         "Contrast": high[outcome].mean() - low[outcome].mean()})
        low_similarity, high_similarity = frame[frame.similarity == 0], frame[frame.similarity == .9]
        for outcome in ("peak_asset_herding", "realised_volatility"):
            rows.append({"Model": label, "Hypothesis": "H2", "Measure": outcome,
                         "Contrast": high_similarity[outcome].mean() - low_similarity[outcome].mean()})
        for outcome in ("realised_volatility", "maximum_drawdown", "normalised_liquidity_deterioration"):
            rows.append({"Model": label, "Hypothesis": "H3", "Measure": outcome,
                         "Contrast": frame.loc[frame.market == "amm", outcome].mean() - frame.loc[frame.market == "traditional", outcome].mean()})
        for outcome in ("realised_volatility", "maximum_drawdown", "recovery_time"):
            rows.append({"Model": label, "Hypothesis": "H4", "Measure": outcome,
                         "Contrast": trad.loc[trad.safeguard == "all", outcome].mean() - trad.loc[trad.safeguard == "none", outcome].mean()})
        return rows
    hypothesis = pd.DataFrame(evidence(original, "original") + evidence(calibrated, "calibrated"))
    hypothesis_path = output_dir / "hypothesis_direction_comparison.csv"
    hypothesis.to_csv(hypothesis_path, index=False)
    return {"ranking": ranking_path, "hypotheses": hypothesis_path}
