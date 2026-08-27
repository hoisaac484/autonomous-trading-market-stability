# Methodology alignment

This file treats the final dissertation draft as the research specification and identifies the corresponding public code, configuration and evidence.

| Dissertation element | Implementation | Reproducible evidence |
|---|---|---|
| Five ETFs, 2010-2025 | \`ASSETS\` in \`model.py\`; frozen Yahoo OHLCV files | \`data/raw/\`, cleaning and quality reports |
| Section 3.3 cleaning | validity filters, common-date alignment, log returns and log volume | \`herding-abm prepare-data\` and \`data/*.csv\` |
| Section 3.4 calibration | return mean/covariance, volume-based relative depth, range-based spread proxy and inverse-depth impact | \`historical_calibration.json\` and \`calibration()\` |
| Historical presentation | descriptive statistics, correlations, 21-day volatility and 60-day rolling correlations | \`data/historical_*\` |
| Trading population | 300 agents; autonomous agents substitute for the background mix | \`simulate()\` and \`configs/default.json\` |
| Background trader types | noise, fundamental and trend-following rules | \`simulate()\` order construction |
| Autonomous agents | multi-asset target weights, momentum/common signals, volatility control and rebalancing | autonomous block in \`simulate()\` |
| Similarity treatment | common-signal weight 0, 0.5 or 0.9 | \`Scenario.similarity\` and \`configs/default.json\` |
| Traditional market | endogenous finite depth, spread and price impact | traditional branch in \`simulate()\` |
| AMM market | constant-product reserves, 0.3% fee, slippage, LP withdrawal and simplified arbitrage | AMM branch in \`simulate()\` |
| Stress design | fundamental, volatility and liquidity shocks | shock branch and config grid |
| Safeguards | position limit, execution delay, circuit breaker, liquidity buffer and combined package | safeguard branches and config grid |
| Primary factorial design | 4 × 3 × 2 × 3 × 6 × 10 | 4,320 rows in \`historical_full_methodology/experiment_runs.csv\` |
| Period analysis | pre-shock, shock and recovery volatility, correlation, depth and price levels | run-level period columns and \`period_contrasts.csv\` |
| H1 | participation and participation × AMM terms; traditional-market participation means | regressions, Figure 5.1 inputs |
| Revised H2 | similarity and participation × similarity; peak asset-level herding | regressions, Figure 5.2 inputs |
| H3 | AMM and participation × AMM; normalised liquidity deterioration | Table 5.2 inputs and regressions |
| H4 | safeguard indicators and relative changes from no safeguard | Figure 5.4 inputs and regressions |
| Validation | 35 historical-simulation target checks plus equal-horizon calibration/holdout comparison | \`outputs/validation/\`, \`outputs/drawdown_calibration/\` |
| Robustness | 13 cases × 2 markets × 3 shocks × 5 seeds | 390 rows in \`outputs/robustness/robustness_runs.csv\` |
| Sensitivity grid | lower impact, deeper liquidity, weaker withdrawal/background flow and heterogeneous reactions | \`configs/calibrated.json\`, 4,320 sensitivity rows and comparison tables |

## Hypothesis wording and scope

- **H1** is a composition result: higher autonomous participation replaces background traders while the total population remains 300. It does not estimate the effect of adding agents to an otherwise unchanged market.
- **H2** tests common-signal similarity and asset-level directional herding. Risk limits and execution rules are not independently varied in this hypothesis.
- **H3** is rejected under the model's reserve, withdrawal and arbitrage assumptions. It is not evidence that real AMMs are generally safer than order-book markets.
- **H4** is partial because the interventions bind strongly in the traditional mechanism but weakly in the AMM averages.

## Deliberate simplifications

The traditional market is not a full limit-order book. Liquidity provision and AMM arbitrage are aggregate mechanisms rather than separately counted agents. The portfolio rule is transparent and heuristic rather than machine learned. Blockchain congestion, gas costs, lending liquidations, individual LP accounting and loss-versus-rebalancing are outside scope.

## Canonical versus sensitivity results

\`configs/default.json\` and \`outputs/historical_full_methodology/\` are the primary specification used for the dissertation's headline results. \`configs/calibrated.json\` and \`outputs/historical_full_calibrated/\` are a sensitivity exercise. The split-sample study improved holdout weighted loss by only 0.6%, below the predeclared 5% adoption threshold, so it did not replace the primary model.

## Random-stream provenance

The configurations explicitly record two random-stream versions. \`primary_v1\` preserves the seeded stream used for the dissertation's primary and robustness grids. \`calibration_v2\` preserves the stream used when the optional reaction-time-heterogeneity calibration was introduced. Within each experiment every treatment is reproducible and uses its stated prespecified seeds. The version field changes random-number allocation only; it does not change an economic equation or parameter. It is retained because silently combining the two historical stream conventions would fail to reproduce the archived 0.6% holdout decision.
