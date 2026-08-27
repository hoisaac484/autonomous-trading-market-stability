from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .model import ASSETS

FIELDS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")


def clean_adjusted_closes(series: dict[str, pd.Series], start: str, end: str) -> tuple[pd.DataFrame, dict]:
    """Align adjusted closes on common dates and return an auditable cleaning report."""
    start_date, end_date = pd.Timestamp(start), pd.Timestamp(end)
    cleaned: dict[str, pd.Series] = {}
    input_rows: dict[str, int] = {}
    duplicate_dates: dict[str, int] = {}
    nonpositive_values: dict[str, int] = {}
    missing_values: dict[str, int] = {}

    for ticker in ASSETS:
        if ticker not in series:
            raise ValueError(f"Missing required ticker: {ticker}")
        values = pd.to_numeric(series[ticker], errors="coerce")
        values.index = pd.to_datetime(values.index).tz_localize(None).normalize()
        input_rows[ticker] = len(values)
        duplicate_dates[ticker] = int(values.index.duplicated(keep="last").sum())
        values = values[~values.index.duplicated(keep="last")].sort_index()
        values = values.loc[(values.index >= start_date) & (values.index <= end_date)]
        nonpositive_values[ticker] = int((values <= 0).fillna(False).sum())
        values = values.where(values > 0)
        missing_values[ticker] = int(values.isna().sum())
        cleaned[ticker] = values

    outer = pd.concat(cleaned, axis=1).sort_index()
    aligned = outer.dropna(how="any")
    if aligned.empty:
        raise ValueError("No common valid trading dates remain after cleaning")
    if len(aligned) < 500:
        raise ValueError(f"Only {len(aligned)} common observations remain; expected at least 500")
    returns = np.log(aligned / aligned.shift(1)).dropna()
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Cleaned prices produce non-finite log returns")

    output = aligned.reset_index().rename(columns={aligned.index.name or "index": "Date"})
    output["Date"] = pd.to_datetime(output["Date"]).dt.strftime("%Y-%m-%d")
    report = {
        "requested_start": start,
        "requested_end": end,
        "actual_start": output["Date"].iloc[0],
        "actual_end": output["Date"].iloc[-1],
        "tickers": list(ASSETS),
        "input_rows": input_rows,
        "duplicate_dates_removed": duplicate_dates,
        "missing_values_detected": missing_values,
        "nonpositive_values_removed": nonpositive_values,
        "outer_union_rows": len(outer),
        "common_aligned_rows": len(output),
        "rows_removed_by_common_date_alignment": len(outer) - len(aligned),
        "return_rows": len(returns),
    }
    return output, report


def download_etf_data(
    output_csv: str | Path,
    start: str = "2010-01-04",
    end: str = "2025-12-31",
    downloader: Callable | None = None,
) -> tuple[Path, Path]:
    """Download Yahoo adjusted closes, preserve raw files, and write aligned data."""
    if downloader is None:
        import yfinance as yf
        downloader = yf.download

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = output_path.parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    # yfinance treats end as exclusive, so request the next calendar day.
    exclusive_end = (date.fromisoformat(end) + timedelta(days=1)).isoformat()
    adjusted: dict[str, pd.Series] = {}
    raw_frames: dict[str, pd.DataFrame] = {}
    for ticker in ASSETS:
        frame = downloader(
            ticker,
            start=start,
            end=exclusive_end,
            interval="1d",
            auto_adjust=False,
            actions=False,
            repair=False,
            progress=False,
            threads=False,
        )
        if frame is None or frame.empty:
            raise RuntimeError(f"Yahoo Finance returned no data for {ticker}")
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        frame.index.name = "Date"
        frame.to_csv(raw_dir / f"{ticker}_yahoo_raw.csv")
        raw_frames[ticker] = frame
        column = "Adj Close" if "Adj Close" in frame.columns else "Close"
        adjusted[ticker] = frame[column].rename(ticker)

    cleaned, report = clean_adjusted_closes(adjusted, start, end)
    cleaned.to_csv(output_path, index=False)
    report_path = output_path.with_suffix(".cleaning_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    build_historical_artifacts(raw_frames, output_path.parent, start, end)
    return output_path, report_path


def build_historical_artifacts(
    frames: dict[str, pd.DataFrame], output_dir: str | Path, start: str, end: str,
    make_plots: bool = True,
) -> dict[str, Path]:
    """Create the complete Section 3.3-3.5 data, audit, and calibration package."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    common_dates: pd.DatetimeIndex | None = None
    prepared: dict[str, pd.DataFrame] = {}
    quality: dict[str, dict] = {}
    for ticker in ASSETS:
        frame = frames[ticker].copy()
        frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
        frame = frame[~frame.index.duplicated(keep="last")].sort_index().loc[start:end]
        for field in FIELDS:
            frame[field] = pd.to_numeric(frame[field], errors="coerce")
        bad_price = frame[["Open", "High", "Low", "Close", "Adj Close"]].le(0).any(axis=1)
        inconsistent_ohlc = ((frame["High"] < frame[["Open", "Close", "Low"]].max(axis=1)) |
                             (frame["Low"] > frame[["Open", "Close", "High"]].min(axis=1)))
        bad_volume = frame["Volume"].le(0)
        missing = frame[list(FIELDS)].isna().any(axis=1)
        valid = ~(bad_price | bad_volume | missing | inconsistent_ohlc)
        quality[ticker] = {
            "input_rows": int(len(frame)), "missing_ohlcv_rows": int(missing.sum()),
            "nonpositive_price_rows": int(bad_price.sum()), "nonpositive_volume_rows": int(bad_volume.sum()),
            "inconsistent_ohlc_rows": int(inconsistent_ohlc.sum()),
            "valid_rows": int(valid.sum()),
        }
        prepared[ticker] = frame.loc[valid, list(FIELDS)]
        common_dates = prepared[ticker].index if common_dates is None else common_dates.intersection(prepared[ticker].index)
    if common_dates is None or len(common_dates) < 500:
        raise ValueError("Insufficient common valid OHLCV observations")

    panels, returns, log_volumes = [], {}, {}
    for ticker in ASSETS:
        frame = prepared[ticker].loc[common_dates].copy()
        frame["Ticker"] = ticker
        frame["LogReturn"] = np.log(frame["Adj Close"] / frame["Adj Close"].shift(1))
        frame["LogVolume"] = np.log(frame["Volume"])
        frame["RangePct"] = (frame["High"] - frame["Low"]) / frame["Close"]
        frame["TurnoverProxy"] = frame["Volume"] * frame["Close"]
        frame["RollingVol21"] = frame["LogReturn"].rolling(21).std() * np.sqrt(252)
        frame.index.name = "Date"
        panels.append(frame.reset_index())
        returns[ticker] = frame["LogReturn"]
        log_volumes[ticker] = frame["LogVolume"]
    panel = pd.concat(panels, ignore_index=True)
    return_frame = pd.DataFrame(returns).dropna()
    log_volume_frame = pd.DataFrame(log_volumes)
    rolling_corr = pd.DataFrame(index=return_frame.index)
    for i, left in enumerate(ASSETS):
        for right in ASSETS[i + 1:]:
            rolling_corr[f"{left}_{right}"] = return_frame[left].rolling(60).corr(return_frame[right])
    panel.to_csv(output_dir / "etf_ohlcv_clean_long.csv", index=False)
    return_frame.reset_index().to_csv(output_dir / "etf_log_returns.csv", index=False)
    log_volume_frame.reset_index().to_csv(output_dir / "etf_log_volume.csv", index=False)
    rolling_corr.reset_index().to_csv(output_dir / "etf_rolling_correlations_60d.csv", index=False)

    rows = []
    extremes = []
    calibration_payload: dict[str, object] = {}
    for ticker in ASSETS:
        subset = panel.loc[panel["Ticker"] == ticker].set_index("Date")
        r = subset["LogReturn"].dropna()
        wealth = np.exp(r.cumsum())
        drawdown = (wealth.cummax() - wealth) / wealth.cummax()
        rows.append({
            "Ticker": ticker, "Observations": len(r), "AnnualisedReturn": r.mean() * 252,
            "AnnualisedVolatility": r.std() * np.sqrt(252), "MinimumReturn": r.min(),
            "Q01": r.quantile(.01), "Q05": r.quantile(.05), "MedianReturn": r.median(),
            "Q95": r.quantile(.95), "Q99": r.quantile(.99), "MaximumReturn": r.max(),
            "Skewness": r.skew(), "ExcessKurtosis": r.kurt(), "Lag1Autocorrelation": r.autocorr(1),
            "AverageVolume": subset["Volume"].mean(), "MedianVolume": subset["Volume"].median(),
            "MaximumDrawdown": drawdown.max(), "MedianRangePct": subset["RangePct"].median(),
        })
        for variable in ("LogReturn", "LogVolume"):
            values = subset[variable].dropna()
            median, mad = values.median(), (values - values.median()).abs().median()
            score = .6745 * (values - median) / max(mad, 1e-12)
            for observed_date in score.abs().nlargest(10).index:
                extremes.append({"Date": observed_date, "Ticker": ticker, "Variable": variable,
                                 "Value": values.loc[observed_date], "RobustZ": score.loc[observed_date],
                                 "Disposition": "Retained; valid positive OHLCV observation"})
    descriptive = pd.DataFrame(rows)
    correlation = return_frame.corr()
    average_volume = panel.groupby("Ticker")["Volume"].mean().reindex(ASSETS)
    depth = np.log1p(average_volume); depth = depth / depth.mean()
    range_proxy = panel.groupby("Ticker")["RangePct"].median().reindex(ASSETS)
    calibration_payload = {
        "assets": list(ASSETS), "daily_mean_returns": return_frame.mean().tolist(),
        "daily_covariance": return_frame.cov().to_numpy().tolist(),
        "annualised_volatility": (return_frame.std() * np.sqrt(252)).tolist(),
        "lag1_return_autocorrelation": [return_frame[a].autocorr(1) for a in ASSETS],
        "relative_depth": depth.tolist(), "baseline_spread_proxy": (range_proxy * .08).tolist(),
        "price_impact_coefficient": (.0032 / depth).tolist(),
        "calibration_notes": {
            "relative_depth": "Normalised log average volume proxy",
            "baseline_spread_proxy": "Eight percent of median daily high-low range; not a quoted-spread estimate",
            "price_impact_coefficient": "Inverse relative-depth mapping anchored at 0.0032",
        },
    }
    descriptive.to_csv(output_dir / "historical_descriptive_statistics.csv", index=False)
    correlation.to_csv(output_dir / "historical_return_correlations.csv")
    pd.DataFrame(extremes).to_csv(output_dir / "extreme_observation_audit.csv", index=False)
    (output_dir / "historical_calibration.json").write_text(json.dumps(calibration_payload, indent=2), encoding="utf-8")
    quality_report = {"start": start, "end": end, "common_rows": len(common_dates), "assets": quality,
                      "extreme_rule": "Ten largest absolute robust-z observations per ticker and variable; retained after OHLCV validity checks"}
    (output_dir / "ohlcv_quality_report.json").write_text(json.dumps(quality_report, indent=2), encoding="utf-8")
    if make_plots:
        _plot_historical_artifacts(panel, rolling_corr, correlation, output_dir)
    return {name: output_dir / name for name in (
        "etf_ohlcv_clean_long.csv", "etf_log_returns.csv", "etf_log_volume.csv",
        "etf_rolling_correlations_60d.csv", "historical_descriptive_statistics.csv",
        "historical_return_correlations.csv", "extreme_observation_audit.csv",
        "historical_calibration.json", "ohlcv_quality_report.json")}


def _plot_historical_artifacts(panel: pd.DataFrame, rolling_corr: pd.DataFrame,
                               correlation: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(11, 5.5))
    for ticker in ASSETS:
        subset = panel.loc[panel["Ticker"] == ticker]
        axis.plot(pd.to_datetime(subset["Date"]), subset["RollingVol21"], label=ticker, linewidth=.9)
    axis.set(title="21-day rolling annualised volatility", ylabel="Annualised volatility", xlabel="Date")
    axis.legend(ncol=5); axis.grid(alpha=.2); fig.tight_layout()
    fig.savefig(output_dir / "historical_rolling_volatility.png", dpi=180); plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 5.5))
    for column in rolling_corr:
        axis.plot(rolling_corr.index, rolling_corr[column], label=column.replace("_", "-"), linewidth=.75)
    axis.set(title="60-day rolling return correlations", ylabel="Correlation", xlabel="Date", ylim=(-1, 1))
    axis.legend(ncol=5, fontsize=7); axis.grid(alpha=.2); fig.tight_layout()
    fig.savefig(output_dir / "historical_rolling_correlations.png", dpi=180); plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.5, 5.5))
    image = axis.imshow(correlation, vmin=-1, vmax=1, cmap="RdBu_r")
    axis.set_xticks(range(len(ASSETS)), ASSETS); axis.set_yticks(range(len(ASSETS)), ASSETS)
    for i in range(len(ASSETS)):
        for j in range(len(ASSETS)):
            axis.text(j, i, f"{correlation.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    axis.set_title("Historical return correlation matrix"); fig.colorbar(image, ax=axis, shrink=.8)
    fig.tight_layout(); fig.savefig(output_dir / "historical_correlation_matrix.png", dpi=180); plt.close(fig)


def rebuild_from_raw(raw_dir: str | Path, output_dir: str | Path, start: str, end: str,
                     make_plots: bool = True) -> dict[str, Path]:
    """Rebuild all frozen historical inputs without making a network request."""
    frames = {}
    for ticker in ASSETS:
        path = Path(raw_dir) / f"{ticker}_yahoo_raw.csv"
        frames[ticker] = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adjusted = {ticker: frame["Adj Close"].rename(ticker) for ticker, frame in frames.items()}
    cleaned, report = clean_adjusted_closes(adjusted, start, end)
    prices_path = output_dir / "etf_adjusted_close_2010_2025.csv"
    report_path = prices_path.with_suffix(".cleaning_report.json")
    cleaned.to_csv(prices_path, index=False)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    artifacts = build_historical_artifacts(frames, output_dir, start, end, make_plots=make_plots)
    return {"prices": prices_path, "cleaning_report": report_path, **artifacts}
