# Historical data

## Source and frozen window

The five files in \`data/raw/\` are daily Yahoo Finance downloads for SPY, QQQ, GLD, TLT and USO covering 4 January 2010 through 31 December 2025. Each contains Date, Open, High, Low, Close, Adjusted Close and Volume. They are committed as a frozen research snapshot so the dissertation can be reproduced without depending on a changing web response.

Yahoo Finance is the data source; the repository is not an official Yahoo data product. Users are responsible for observing the source's terms when redistributing or refreshing data.

## Cleaning rules

\`herding-abm prepare-data\` performs the following deterministic transformations:

1. Parse dates, remove duplicate dates by retaining the last observation, and sort chronologically.
2. Restrict every series to the stated window.
3. Reject missing or non-positive OHLCV observations and inconsistent high/low rows.
4. Intersect the five valid trading-date indexes; no forward filling is used.
5. Calculate adjusted-close log returns, log volume, high-low range, turnover proxy and 21-day rolling annualised volatility.
6. Calculate 60-day rolling pairwise return correlations.
7. Retain valid extreme returns/volumes while recording the ten largest absolute robust-z observations per ticker and variable.
8. Derive descriptive statistics and transparent liquidity calibration proxies.

The frozen snapshot contains 4,024 common valid dates and 4,023 return observations. The reports \`etf_adjusted_close_2010_2025.cleaning_report.json\` and \`ohlcv_quality_report.json\` record the audit counts.

## Exact rebuild

\`\`\`bash
herding-abm prepare-data --raw data/raw --output reproduced-data
\`\`\`

Use \`--no-plots\` in a minimal numerical environment. This changes only the optional PNG generation, not any CSV or JSON result.
