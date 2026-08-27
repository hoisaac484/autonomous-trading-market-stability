from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ASSETS = ("SPY", "QQQ", "GLD", "TLT", "USO")


@dataclass(frozen=True)
class Scenario:
    market: str = "traditional"
    participation: float = 0.5
    similarity: float = 0.5
    shock: str = "fundamental"
    safeguard: str = "none"
    seed: int = 11


def calibration(prices_csv: str | None, seed: int = 7, start: str | None = None,
                end: str | None = None) -> dict[str, np.ndarray]:
    """Estimate drift/covariance from aligned adjusted closes or a stable proxy sample."""
    if prices_csv:
        frame = pd.read_csv(Path(prices_csv), parse_dates=["Date"]).set_index("Date")
        prices = frame.loc[:, ASSETS].dropna()
        if start or end:
            prices = prices.loc[start:end]
        returns = np.log(prices / prices.shift(1)).dropna()
    else:
        rng = np.random.default_rng(seed)
        vols = np.array([0.011, 0.014, 0.010, 0.009, 0.022])
        corr = np.array([
            [1.00, .86, .08, -.20, .20], [.86, 1.00, .04, -.22, .18],
            [.08, .04, 1.00, .15, .12], [-.20, -.22, .15, 1.00, -.08],
            [.20, .18, .12, -.08, 1.00],
        ])
        returns = pd.DataFrame(
            rng.multivariate_normal(np.array([.00035, .00045, .00025, .00010, .00015]),
                                    np.outer(vols, vols) * corr, size=4030),
            columns=ASSETS,
        )
    result = {"mu": returns.mean().to_numpy(), "cov": returns.cov().to_numpy()}
    calibration_file = Path(prices_csv).parent / "historical_calibration.json" if prices_csv else None
    panel_file = Path(prices_csv).parent / "etf_ohlcv_clean_long.csv" if prices_csv else None
    if panel_file and panel_file.exists() and (start or end):
        panel = pd.read_csv(panel_file, parse_dates=["Date"])
        panel = panel.loc[(panel["Date"] >= pd.Timestamp(start or panel["Date"].min())) &
                          (panel["Date"] <= pd.Timestamp(end or panel["Date"].max()))]
        average_volume = panel.groupby("Ticker")["Volume"].mean().reindex(ASSETS)
        depth = np.log1p(average_volume); depth = depth / depth.mean()
        ranges = panel.groupby("Ticker")["RangePct"].median().reindex(ASSETS)
        result.update({"depth": depth.to_numpy(), "spread": (ranges * .08).to_numpy(),
                       "impact": (.0032 / depth).to_numpy()})
    elif calibration_file and calibration_file.exists():
        payload = __import__("json").loads(calibration_file.read_text(encoding="utf-8"))
        result.update({"depth": np.asarray(payload["relative_depth"]),
                       "spread": np.asarray(payload["baseline_spread_proxy"]),
                       "impact": np.asarray(payload["price_impact_coefficient"])})
    else:
        result.update({"depth": np.array([1.25, 1.0, .75, .8, .6]),
                       "spread": np.ones(len(ASSETS)) * .0004,
                       "impact": np.ones(len(ASSETS)) * .0032})
    return result


def _max_drawdown(prices: np.ndarray) -> np.ndarray:
    peaks = np.maximum.accumulate(prices, axis=0)
    return np.max((peaks - prices) / peaks, axis=0)


def _recovery(series: np.ndarray, shock_start: int, shock_end: int) -> float:
    baseline = float(np.mean(series[max(0, shock_start - 30):shock_start]))
    tolerance = max(abs(baseline) * .20, 1e-8)
    for offset, value in enumerate(series[shock_end:]):
        if abs(float(value) - baseline) <= tolerance:
            return float(offset)
    return float(len(series) - shock_end)


def _period_metrics(returns: np.ndarray, depth: np.ndarray, prices: np.ndarray, start: int, end: int, label: str) -> dict[str, float]:
    block = returns[start:end]
    corr = np.corrcoef(block.T) if len(block) > 2 else np.eye(block.shape[1])
    return {
        f"{label}_volatility": float(np.sqrt(np.sum(block ** 2, axis=0)).mean()),
        f"{label}_correlation": float(corr[np.triu_indices(block.shape[1], 1)].mean()),
        f"{label}_depth": float(depth[start:end].mean()),
        f"{label}_price_level": float(prices[start:end].mean()),
    }


def simulate(config: dict[str, Any], scenario: Scenario, calib: dict[str, np.ndarray]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run one controlled market path with comparable agent logic across mechanisms."""
    rng = np.random.default_rng(scenario.seed)
    steps, n_assets = int(config["steps"]), len(ASSETS)
    shock_start = int(config["shock_start"])
    shock_end = shock_start + int(config["shock_duration"])
    n_agents = int(config["n_agents"])
    n_auto = int(round(n_agents * scenario.participation))
    n_noise = int((n_agents - n_auto) * float(config.get("noise_share", .45)))
    n_fund = int((n_agents - n_auto) * float(config.get("fundamental_share", .35)))
    n_trend = max(0, n_agents - n_auto - n_noise - n_fund)

    prices = np.ones((steps, n_assets)) * 100.0
    fundamental = np.ones((steps, n_assets)) * 100.0
    returns = np.zeros((steps, n_assets))
    order_flow = np.zeros((steps, n_assets))
    depth = np.ones((steps, n_assets))
    spreads = np.zeros((steps, n_assets))
    slippage = np.zeros((steps, n_assets))
    herding = np.zeros(steps)
    asset_herding = np.zeros((steps, n_assets))
    common_signal = rng.normal(0, 1, (steps, n_assets))
    idio_parameters = rng.normal(1, .25, (max(n_auto, 1), n_assets))
    reaction_sd = float(config.get("reaction_time_heterogeneity", 0.0))
    extended_stream = config.get("random_stream_version", "primary_v1") == "calibration_v2"
    # Do not advance the random stream when heterogeneity is disabled.  The
    # dissertation baseline predates the optional heterogeneity extension and
    # this preserves exact seeded reproduction of the primary 4,320-run grid.
    autonomous_reaction = (np.clip(rng.normal(1, reaction_sd, max(n_auto, 1)), .1, 1.9)
                           if reaction_sd > 0 or extended_stream else np.ones(max(n_auto, 1)))
    wealth = np.ones(max(n_auto, 1))
    weights = np.tile(np.array([.30, .25, .15, .15, .15]), (max(n_auto, 1), 1))
    reserves_x = np.ones(n_assets) * 1200.0
    reserves_y = reserves_x * 100.0
    base_depth = np.asarray(calib["depth"]) * float(config.get("depth_multiplier", 1.0))
    lp_capacity = base_depth.copy()
    circuit_halt = 0
    delayed_orders = np.zeros(n_assets)
    log_volatility_state = 0.0
    intervals_per_day = float(config.get("intervals_per_day", 24))
    safeguards = {scenario.safeguard} if scenario.safeguard != "all" else {
        "execution_delay", "position_limit", "liquidity_buffer", "circuit_breaker"
    }

    for t in range(1, steps):
        # Heavy-tailed innovations with persistent stochastic volatility preserve the
        # daily calibration when aggregated across the configured intraday intervals.
        log_volatility_state = .975 * log_volatility_state + rng.normal(0, .035)
        gaussian = rng.multivariate_normal(np.zeros(n_assets), calib["cov"] / intervals_per_day)
        tail_scale = np.sqrt(4 / rng.chisquare(6))
        innovation = calib["mu"] / intervals_per_day + gaussian * tail_scale * np.exp(log_volatility_state)
        volatility_multiplier = 1.0
        liquidity_multiplier = 1.0
        if shock_start <= t < shock_end:
            if scenario.shock == "fundamental":
                innovation[0] -= .006 * float(config.get("shock_magnitude_multiplier", 1.0))
            elif scenario.shock == "volatility":
                volatility_multiplier = 1 + 2 * float(config.get("shock_magnitude_multiplier", 1.0))
                innovation += rng.multivariate_normal(np.zeros(n_assets), calib["cov"] / 12) * float(config.get("shock_magnitude_multiplier", 1.0))
            elif scenario.shock == "liquidity":
                liquidity_multiplier = max(.05, 1 - .72 * float(config.get("shock_magnitude_multiplier", 1.0)))
        fundamental[t] = fundamental[t - 1] * np.exp(innovation)

        long_window = int(config.get("long_signal_window", 20))
        short_window = int(config.get("short_signal_window", 5))
        history = returns[max(0, t - long_window):t]
        recent = returns[max(0, t - short_window):t].mean(axis=0)
        long = history.mean(axis=0)
        vol = history.std(axis=0) if len(history) > 2 else np.sqrt(np.diag(calib["cov"]))

        noise_orders = rng.normal(0, .0025, (n_noise, n_assets)).sum(axis=0)
        mispricing = (fundamental[t] - prices[t - 1]) / prices[t - 1]
        background_multiplier = float(config.get("background_order_multiplier", 1.0))
        if reaction_sd > 0 or extended_stream:
            fundamental_activation = np.clip(rng.normal(1, reaction_sd, n_assets), .1, 1.9)
            trend_activation = np.clip(rng.normal(1, reaction_sd, n_assets), .1, 1.9)
        else:
            fundamental_activation = trend_activation = np.ones(n_assets)
        noise_orders *= background_multiplier
        fundamental_orders = background_multiplier * n_fund * .012 * mispricing * fundamental_activation
        trend_orders = background_multiplier * n_trend * .007 * np.tanh((recent - long) * 80) * trend_activation

        auto_orders = np.zeros((max(n_auto, 1), n_assets))
        if n_auto:
            signal = scenario.similarity * common_signal[t] + (1 - scenario.similarity) * rng.normal(0, 1, (n_auto, n_assets))
            expected = .45 * recent + .15 * signal * np.sqrt(np.diag(calib["cov"]))
            risk_scale = np.minimum(1.0, .011 / np.maximum(vol.mean() * (1 + .12 * signal.mean(axis=1)), .001))
            raw_target = np.exp(expected * 40) * idio_parameters[:n_auto]
            raw_target /= raw_target.sum(axis=1, keepdims=True)
            target = raw_target * risk_scale[:, None]
            auto_orders[:n_auto] = (.16 * float(config.get("rebalancing_speed_multiplier", 1.0)) *
                                    autonomous_reaction[:n_auto, None] *
                                    (target - weights[:n_auto]) * wealth[:n_auto, None])
            if "position_limit" in safeguards:
                auto_orders[:n_auto] = np.clip(auto_orders[:n_auto], -.012, .012)
            if "execution_delay" in safeguards:
                active = rng.random(n_auto) > .5
                auto_orders[:n_auto] *= active[:, None]
            directions = np.sign(auto_orders[:n_auto].sum(axis=1))
            buys, sells = np.sum(directions > 0), np.sum(directions < 0)
            herding[t] = abs(buys - sells) / max(buys + sells, 1)
            for i in range(n_assets):
                asset_directions = np.sign(auto_orders[:n_auto, i])
                ab, ass = np.sum(asset_directions > 0), np.sum(asset_directions < 0)
                asset_herding[t, i] = abs(ab - ass) / max(ab + ass, 1)

        net = noise_orders + fundamental_orders + trend_orders + auto_orders[:n_auto].sum(axis=0)
        net += delayed_orders
        delayed_orders[:] = 0
        if circuit_halt:
            delayed_orders += net
            net[:] = 0
            circuit_halt -= 1

        if scenario.market == "traditional":
            risk_tolerance = max(float(config.get("lp_risk_tolerance_multiplier", 1.0)), .1)
            withdrawal_strength = float(config.get("liquidity_withdrawal_strength", 1.0))
            stress = 1 + withdrawal_strength * (12 * vol + .25 * np.abs(net)) / risk_tolerance
            lp_capacity = .92 * lp_capacity + .08 * base_depth / stress
            lp_capacity *= liquidity_multiplier
            if "liquidity_buffer" in safeguards:
                lp_capacity = np.maximum(lp_capacity, base_depth * .55)
            depth[t] = lp_capacity
            spreads[t] = calib["spread"] + .035 * vol + .0008 * np.abs(net) / np.maximum(lp_capacity, .05)
            impact = (float(config.get("price_impact_multiplier", 1.0)) * calib["impact"] *
                      net / np.maximum(lp_capacity, .05))
            market_return = innovation + impact + rng.normal(0, .00015 * volatility_multiplier, n_assets)
        else:
            if scenario.shock == "liquidity" and shock_start <= t < shock_end:
                reserves_x *= .985
                reserves_y *= .985
            adverse = np.maximum(vol.mean() - .012, 0)
            withdrawal = min(.012, adverse * .25)
            if "liquidity_buffer" in safeguards:
                withdrawal = min(withdrawal, .002)
            reserves_x *= 1 - withdrawal
            reserves_y *= 1 - withdrawal
            old_price = reserves_y / reserves_x
            fee = .003
            for i, q in enumerate(net):
                quote = q * old_price[i] * 4.0
                if quote >= 0:
                    y_in = min(quote * (1 - fee), reserves_y[i] * .20)
                    k = reserves_x[i] * reserves_y[i]
                    reserves_y[i] += y_in
                    reserves_x[i] = k / reserves_y[i]
                else:
                    x_in = min((-quote / old_price[i]) * (1 - fee), reserves_x[i] * .20)
                    k = reserves_x[i] * reserves_y[i]
                    reserves_x[i] += x_in
                    reserves_y[i] = k / reserves_x[i]
            amm_price = reserves_y / reserves_x
            gap = np.log(fundamental[t] / amm_price)
            arb = np.where(np.abs(gap) > .003, .35 * gap, 0)
            amm_price *= np.exp(arb)
            reserves_y = reserves_x * amm_price
            market_return = np.log(amm_price / prices[t - 1])
            slippage[t] = np.abs(np.log(np.maximum(amm_price, 1e-9) / np.maximum(old_price, 1e-9)))
            depth[t] = np.sqrt(reserves_x * reserves_y) / np.sqrt(120000.0)
            spreads[t] = fee + slippage[t]

        if "circuit_breaker" in safeguards and np.max(np.abs(market_return)) > .045:
            market_return = np.clip(market_return, -.045, .045)
            circuit_halt = 2
        returns[t] = market_return
        prices[t] = prices[t - 1] * np.exp(market_return)
        order_flow[t] = net
        if n_auto:
            weights[:n_auto] = np.clip(weights[:n_auto] + auto_orders[:n_auto], 0, 1)
            weights[:n_auto] /= np.maximum(weights[:n_auto].sum(axis=1, keepdims=True), 1e-9)
            wealth[:n_auto] *= np.exp(returns[t].mean())

    records: dict[str, np.ndarray] = {"step": np.arange(steps), "herding": herding}
    for i, asset in enumerate(ASSETS):
        records[f"price_{asset}"] = prices[:, i]
        records[f"return_{asset}"] = returns[:, i]
        records[f"depth_{asset}"] = depth[:, i]
        records[f"spread_{asset}"] = spreads[:, i]
        records[f"flow_{asset}"] = order_flow[:, i]
        records[f"slippage_{asset}"] = slippage[:, i]
        records[f"asset_herding_{asset}"] = asset_herding[:, i]
    frame = pd.DataFrame(records)
    shock_slice = slice(shock_start, min(steps, shock_end + 40))
    corr = np.corrcoef(returns[shock_slice].T)
    metrics: dict[str, Any] = {
        **scenario.__dict__,
        "realised_volatility": float(np.sqrt(np.sum(returns[shock_slice] ** 2, axis=0)).mean()),
        "maximum_drawdown": float(_max_drawdown(prices).mean()),
        "mean_pairwise_correlation": float(corr[np.triu_indices(n_assets, 1)].mean()),
        "peak_herding": float(herding[shock_slice].max()),
        "peak_asset_herding": float(asset_herding[shock_slice].max()),
        "mean_depth": float(depth[shock_slice].mean()),
        "mean_spread_or_cost": float(spreads[shock_slice].mean()),
        "mean_absolute_order_flow": float(np.abs(order_flow[shock_slice]).mean()),
        "normalised_liquidity_deterioration": float(1 - depth[shock_start:shock_end].mean() / max(depth[max(0, shock_start-30):shock_start].mean(), 1e-9)),
        "recovery_time": _recovery(np.mean(np.abs(returns), axis=1), shock_start, shock_end),
        "price_recovery_time": _recovery(np.mean(prices, axis=1), shock_start, shock_end),
        "volatility_recovery_time": _recovery(pd.Series(np.mean(np.abs(returns), axis=1)).rolling(10, min_periods=1).mean().to_numpy(), shock_start, shock_end),
        "liquidity_recovery_time": _recovery(np.mean(depth, axis=1), shock_start, shock_end),
    }
    metrics.update(_period_metrics(returns, depth, prices, max(1, shock_start - 40), shock_start, "pre"))
    metrics.update(_period_metrics(returns, depth, prices, shock_start, min(shock_end, steps), "shock"))
    metrics.update(_period_metrics(returns, depth, prices, min(shock_end, steps - 1), min(steps, shock_end + 40), "recovery"))
    return frame, metrics
