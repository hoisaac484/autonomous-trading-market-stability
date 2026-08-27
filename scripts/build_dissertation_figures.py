"""Regenerate the three Chapter 5 figures retained in the dissertation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


COLOURS = {"volatility": "#2A7F8E", "drawdown": "#D89A2B", "correlation": "#6B7280"}


def finish(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path,
                        default=Path("outputs/historical_full_methodology/experiment_runs.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/dissertation_figures"))
    args = parser.parse_args()
    data = pd.read_csv(args.runs)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    # Figure 5.1: H1 in the fixed-population traditional market.
    h1 = data[data.market.eq("traditional")].groupby("participation")[[
        "realised_volatility", "maximum_drawdown", "mean_pairwise_correlation"
    ]].mean()
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0))
    for axis, column, title, colour in zip(
        axes, h1.columns, ("Realised volatility", "Maximum drawdown", "Pairwise correlation"),
        COLOURS.values(),
    ):
        axis.plot(h1.index * 100, h1[column], marker="o", linewidth=2, color=colour)
        axis.set(title=title, xlabel="Autonomous participation (%)")
        axis.grid(axis="y", alpha=.25)
    finish(fig, args.output / "figure_5_1_h1_participation.png")

    # Figure 5.2: H2 primary outcome plus indexed secondary outcomes.
    h2 = data.groupby("similarity")[["peak_asset_herding", "realised_volatility",
                                     "maximum_drawdown", "mean_pairwise_correlation"]].mean()
    indexed = h2[["realised_volatility", "maximum_drawdown", "mean_pairwise_correlation"]]
    indexed = indexed.div(indexed.iloc[0]).mul(100)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
    axes[0].plot(h2.index, h2.peak_asset_herding, marker="o", linewidth=2, color=COLOURS["volatility"])
    axes[0].set(title="Peak asset-level directional herding", xlabel="Common-signal similarity")
    for column, colour in zip(indexed, COLOURS.values()):
        axes[1].plot(indexed.index, indexed[column], marker="o", linewidth=2,
                     label=column.replace("_", " ").title(), color=colour)
    axes[1].axhline(100, color="black", linewidth=.8, alpha=.5)
    axes[1].set(title="Secondary outcomes (similarity 0 = 100)", xlabel="Common-signal similarity")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=.25)
    finish(fig, args.output / "figure_5_2_h2_similarity.png")

    # Figure 5.4: percentage change in volatility from each market's no-safeguard mean.
    safeguard = data.groupby(["market", "safeguard"])["realised_volatility"].mean().unstack()
    reductions = (1 - safeguard.div(safeguard["none"], axis=0)).mul(100)
    order = ["position_limit", "execution_delay", "circuit_breaker", "liquidity_buffer", "all"]
    chart = reductions[order].T
    fig, axis = plt.subplots(figsize=(8.4, 3.6))
    chart.plot(kind="bar", ax=axis, color=["#2A7F8E", "#D89A2B"])
    axis.set(ylabel="Reduction in realised volatility (%)", xlabel="Safeguard")
    axis.set_xticklabels([name.replace("_", " ").title() for name in order], rotation=0)
    axis.legend(title="Market", frameon=False)
    axis.grid(axis="y", alpha=.25)
    finish(fig, args.output / "figure_5_4_h4_safeguards.png")


if __name__ == "__main__":
    main()
