from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PRIMARY_FACTORS = {
    "market": {"traditional", "amm"},
    "participation": {0.0, 0.25, 0.5, 0.75},
    "similarity": {0.0, 0.5, 0.9},
    "shock": {"fundamental", "volatility", "liquidity"},
    "safeguard": {"none", "execution_delay", "position_limit", "liquidity_buffer",
                  "circuit_breaker", "all"},
    "seed": {11, 29, 47, 71, 101, 131, 173, 211, 257, 307},
}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(root: str | Path, manifest_path: str | Path) -> list[str]:
    root = Path(root)
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    errors: list[str] = []
    for relative, expected in payload["sha256"].items():
        path = root / relative
        if not path.exists():
            errors.append(f"Missing manifest file: {relative}")
        elif sha256(path) != expected:
            errors.append(f"Hash mismatch: {relative}")
    return errors


def verify_primary_results(runs_csv: str | Path) -> dict[str, object]:
    runs = pd.read_csv(runs_csv)
    errors: list[str] = []
    if len(runs) != 4320:
        errors.append(f"Expected 4,320 primary runs; found {len(runs):,}")
    for column, expected in PRIMARY_FACTORS.items():
        actual = set(runs[column].unique())
        if actual != expected:
            errors.append(f"Unexpected {column} levels: {sorted(actual, key=str)}")
    keys = ["market", "participation", "similarity", "shock", "safeguard"]
    counts = runs.groupby(keys).size()
    if len(counts) != 432 or not counts.eq(10).all():
        errors.append("The primary grid must contain 432 cells with 10 seeds in every cell")
    numeric = runs.select_dtypes(include="number")
    if not np.isfinite(numeric.to_numpy()).all():
        errors.append("Primary results contain non-finite numeric values")

    traditional = runs.loc[runs["market"].eq("traditional")]
    h1 = traditional.groupby("participation")[[
        "realised_volatility", "maximum_drawdown", "mean_pairwise_correlation"
    ]].mean()
    h2 = runs.groupby("similarity")["peak_asset_herding"].mean()
    markets = runs.groupby("market")[[
        "realised_volatility", "maximum_drawdown", "mean_pairwise_correlation",
        "normalised_liquidity_deterioration",
    ]].mean()
    safeguards = traditional.groupby("safeguard")["realised_volatility"].mean()
    all_reduction = 1 - safeguards["all"] / safeguards["none"]
    headline = {
        "h1_volatility_at_0": float(h1.loc[0.0, "realised_volatility"]),
        "h1_volatility_at_75": float(h1.loc[0.75, "realised_volatility"]),
        "h1_drawdown_at_0": float(h1.loc[0.0, "maximum_drawdown"]),
        "h1_drawdown_at_75": float(h1.loc[0.75, "maximum_drawdown"]),
        "h1_correlation_at_0": float(h1.loc[0.0, "mean_pairwise_correlation"]),
        "h1_correlation_at_75": float(h1.loc[0.75, "mean_pairwise_correlation"]),
        "h2_herding_at_0": float(h2.loc[0.0]),
        "h2_herding_at_09": float(h2.loc[0.9]),
        "h4_all_traditional_volatility_reduction": float(all_reduction),
        "amm_lower_on_all_h3_outcomes": bool((markets.loc["amm"] < markets.loc["traditional"]).all()),
    }
    expected = {
        "h1_volatility_at_0": 0.0439,
        "h1_volatility_at_75": 0.1140,
        "h1_drawdown_at_0": 0.105,
        "h1_drawdown_at_75": 0.246,
        "h1_correlation_at_0": 0.118,
        "h1_correlation_at_75": 0.231,
        "h2_herding_at_0": 0.584,
        "h2_herding_at_09": 0.750,
        "h4_all_traditional_volatility_reduction": 0.779,
    }
    for name, target in expected.items():
        if not np.isclose(headline[name], target, atol=0.001):
            errors.append(f"Draft headline mismatch for {name}: {headline[name]:.6f} vs {target:.6f}")
    if not headline["amm_lower_on_all_h3_outcomes"]:
        errors.append("H3 direction does not match the dissertation: AMM outcomes are not all lower")
    return {"ok": not errors, "errors": errors, "headline": headline}
