# Algorithmic Herding and Market Volatility

Reproducible code and frozen research inputs for a dissertation comparing rule-based autonomous portfolio-rebalancing agents in a stylised traditional market and a constant-product automated market maker (AMM).

The agents are transparent rule-based portfolio systems. They are not high-frequency traders, learning agents, generative-AI systems, or a claim about artificial intelligence in general.

## Dissertation design

The primary experiment contains **4,320 runs**:

```text
4 participation levels × 3 similarity levels × 2 markets
× 3 shocks × 6 safeguard settings × 10 seeds = 4,320
```

- Assets: SPY, QQQ, GLD, TLT and USO.
- Frozen historical window: 4 January 2010 to 31 December 2025.
- 300 trading agents in each primary run.
- 500 intervals per run; shock begins at interval 200 and lasts 40 intervals.
- 24 intervals are mapped to one trading day for annualisation.
- Fixed-population substitution: increasing the autonomous share proportionally replaces noise, fundamental and trend-following traders.
- Separate 390-run robustness grid and split-sample calibration/holdout exercise.

The exact mapping from the dissertation to code and outputs is in [docs/methodology-alignment.md](docs/methodology-alignment.md).

## Install

Python 3.12 is the reference environment. The numerical dependencies are pinned because seeded simulation paths can change across library versions.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,plots]"
```

## Verify the archived dissertation results

This fast check validates the frozen inputs, the 4,320-row factorial structure, finite outputs, and the headline H1-H4 values reported in the dissertation:

```bash
herding-abm verify
python -m unittest discover -s tests -v
```

Expected primary conclusions:

- H1 supported within the fixed-population traditional-market design.
- Revised H2 supported for asset-level directional herding.
- H3 not supported: under the stated parameterisation, the AMM has lower primary instability averages.
- H4 partially supported: safeguards strongly reduce traditional-market volatility but are weaker in the AMM.

## Reproduce from frozen raw data

Run a 96-path smoke test first. It rebuilds the cleaned historical dataset from the five frozen raw CSV files and writes into a new directory without modifying the archived results:

```bash
python scripts/reproduce.py --mode quick --output reproduced-quick --no-plots
```

Run the complete dissertation workflow:

```bash
python scripts/reproduce.py --mode full --output reproduced-full
```

The full mode rebuilds the data transformations, primary 4,320-run experiment, analysis, baseline validation, 390-run robustness grid, split-sample drawdown calibration, alternative 4,320-run sensitivity grid, and ranking comparison. Runtime depends on hardware and is substantially longer than the quick check.

To regenerate the three Chapter 5 figures retained in the final draft:

```bash
python scripts/build_dissertation_figures.py \
  --runs outputs/historical_full_methodology/experiment_runs.csv \
  --output reproduced-figures
```

## Data provenance

The committed raw files are a frozen daily Yahoo Finance OHLCV snapshot. Exact reproduction uses this snapshot and makes no network request. `prepare-data` performs validity checks, common-date alignment, log-return and log-volume transformations, rolling statistics, descriptive statistics, liquidity proxies and an extreme-observation audit.

To deliberately refresh the source data, install the download extra and run:

```bash
python -m pip install -e ".[download,plots]"
herding-abm download-data --output data/etf_adjusted_close_2010_2025.csv
```

A refreshed download is a new dataset and is not expected to have the frozen-input hashes in `reproducibility/manifest.json`. See [data/README.md](data/README.md).

## Repository layout

```text
configs/                 Primary, quick, robustness and sensitivity specifications
data/raw/                Frozen Yahoo Finance inputs
data/                    Rebuilt historical transformations and calibration targets
docs/                    Methodology-to-code alignment and interpretation boundaries
outputs/                 Curated numerical results used in the dissertation
reproducibility/         Input/result checksums
scripts/                 End-to-end reproduction and figure generation
src/herding_abm/         Model, data pipeline, analysis and command-line interface
tests/                   Determinism, design and data-cleaning tests
```

## Interpretation boundary

The traditional mechanism is a finite-depth aggregate price-impact model, not a reconstructed exchange limit-order book. AMM pools use a cash numeraire, simplified liquidity-provider withdrawal and arbitrage. The results support controlled comparisons between scenarios; absolute simulated losses are not forecasts of future ETF or DeFi losses.
