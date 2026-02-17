# Statistical Arbitrage Research Pipeline (Pairs Trading)

A research-oriented statistical arbitrage pipeline for pairs trading, built for correctness and rigor.

## Features

- **Data ingestion** — Yahoo Finance or CSV, with flexible date ranges
- **Cointegration-based pairs trading** — Engle-Granger hedge ratio with OLS or Kalman filter
- **Z-score mean reversion signals** — lookahead-safe execution (1-day signal shift)
- **Walk-forward backtesting** — daily parameter refit with transaction costs and slippage
- **Statistical validation** — Sharpe, drawdown, CAGR, bootstrap confidence intervals, FDR-controlled selection
- **ML pair selection** — OPTICS/DBSCAN clustering as an alternative to correlation prefiltering
- **Advanced risk management** — Ledoit-Wolf covariance, correlation-aware weights, position limits, drawdown breaker
- **Live trading framework** — paper broker with slippage/fee simulation, signal generation, order execution
- **Visualization** — equity curves, drawdown charts, spread/z-score plots, rolling beta, correlation heatmaps
- **EDA notebook** — interactive Jupyter walkthrough of the full pipeline

## Strategy Overview

We model a pair (Y, X) with a linear hedge ratio on the training window:

```
Y_t ≈ α + β X_t
Spread: s_t = Y_t - (α + β X_t)
Z-score: z_t = (s_t - rolling_mean) / rolling_std
```

Rules:
- Enter long spread when `z < -entry_z`
- Enter short spread when `z > +entry_z`
- Exit when `|z| < exit_z`

Signals are shifted by 1 day to avoid lookahead bias. Hedge ratios can be estimated via OLS or an online Kalman filter.

## Project Structure

```
src/sarb/
├── data/           # Data ingestion (yfinance, CSV)
│   └── ingest.py
├── features/       # Spread computation & hedge ratios
│   ├── spread.py       # OLS hedge ratio, spread, rolling z-score
│   └── kalman.py       # Online Kalman filter hedge ratio
├── split/          # Time-aware data splitting
│   ├── time_split.py   # Train/val/test split
│   └── rebalance.py    # Rolling quarterly windows
├── strategy/       # Signal generation
│   └── pairs.py        # Z-score threshold positions
├── backtest/       # Backtesting engines
│   ├── engine.py       # Single-pair backtest with costs + borrow fees
│   └── walkforward.py  # Walk-forward daily refit (OLS or Kalman)
├── metrics/        # Performance measurement
│   └── performance.py  # Sharpe, max drawdown, CAGR
├── stats/          # Statistical tests
│   ├── cointegration.py    # ADF test, half-life estimation
│   ├── multiple_testing.py # Benjamini-Hochberg FDR control
│   └── bootstrap.py        # Bootstrap confidence intervals
├── research/       # Research pipeline
│   ├── select_pairs.py         # Pair scanning with FDR control
│   ├── walkforward_portfolio.py # Multi-pair quarterly portfolio
│   └── ml_select.py            # OPTICS/DBSCAN pair clustering
├── portfolio/      # Portfolio construction
│   └── vol_target.py  # Volatility targeting & scaling
├── risk/           # Risk management
│   ├── limits.py       # Position limits, drawdown breaker, borrow costs
│   └── covariance.py   # Ledoit-Wolf shrinkage, correlation-aware weights
├── live/           # Live trading framework
│   ├── broker.py       # Abstract broker interface (Order, Fill, BaseBroker)
│   ├── paper_broker.py # Paper trading broker with slippage/fees
│   ├── signal.py       # Live signal generation (PairSignal)
│   └── runner.py       # Live execution loop (LiveConfig, run_live_step)
├── viz/            # Visualization
│   ├── charts.py       # Equity, drawdown, spread, z-score, heatmap plots
│   └── report.py       # Automated backtest report generation
└── config.py       # Global configuration

scripts/
├── run_pairs.py                  # Single-pair backtest with visualization
├── run_walkforward_portfolio.py  # Multi-pair portfolio (3 modes: equal weight, vol-target, risk-managed)
├── scan_pairs.py                 # Universe scanning with FDR-controlled selection
└── run_live.py                   # Paper trading simulation

notebooks/
└── 01_eda_pairs.ipynb  # EDA: prices, correlations, spreads, cointegration, backtest

tests/                  # 59 unit tests (all synthetic data, no network)
├── conftest.py         # Shared fixtures (synthetic cointegrated prices)
├── test_data.py        # CSV ingestion
├── test_features.py    # Hedge ratio, spread, z-score
├── test_kalman.py      # Kalman filter convergence
├── test_split.py       # Time splits, rolling windows
├── test_strategy.py    # Position entry/exit logic
├── test_backtest.py    # Equity, cost validation
├── test_metrics.py     # Sharpe, drawdown, CAGR
├── test_stats.py       # ADF, half-life, FDR, bootstrap
├── test_research.py    # Pair evaluation, scanning
├── test_portfolio.py   # Vol targeting, realized vol
├── test_ml_select.py   # OPTICS/DBSCAN clustering
├── test_risk.py        # Shrinkage, limits, drawdown breaker
├── test_viz.py         # Chart generation, file saving
└── test_live.py        # Paper broker, signals, live runner
```

## Installation

```bash
# Core install
pip install -e .

# With all optional dependencies (visualization, ML, testing)
pip install -e ".[all]"

# Individual extras
pip install -e ".[viz]"   # matplotlib for charts
pip install -e ".[ml]"    # scikit-learn for ML pair selection
pip install -e ".[dev]"   # pytest for testing
```

**Requirements:** Python 3.10+, numpy, pandas, yfinance, statsmodels

## Quickstart

### Single-pair backtest

```bash
python scripts/run_pairs.py
```

Runs a walk-forward backtest on KO/PEP with daily OLS refit, prints Sharpe/drawdown/CAGR, and saves plots to `reports/figures/`.

### Multi-pair portfolio

```bash
python scripts/run_walkforward_portfolio.py
```

Runs three portfolio modes on a 6-ticker universe:
1. **Equal weight** — baseline with no risk adjustments
2. **Vol-targeted** — volatility scaling to target daily vol
3. **Risk-managed** — correlation-aware weights + position limits + drawdown breaker

### Universe scanning

```bash
python scripts/scan_pairs.py
```

Scans all ticker combinations, applies ADF cointegration test with Benjamini-Hochberg FDR control, and ranks pairs by half-life.

### Paper trading simulation

```bash
python scripts/run_live.py
```

Simulates daily paper trading on KO/PEP and XOM/CVX with a $100K account, printing periodic account value updates.

### EDA notebook

```bash
jupyter notebook notebooks/01_eda_pairs.ipynb
```

Interactive walkthrough: data loading, normalized prices, correlation heatmaps, spread analysis, cointegration diagnostics, and walk-forward backtest results.

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Or in two batches (faster on some systems)
python -m pytest tests/test_metrics.py tests/test_data.py tests/test_features.py tests/test_split.py tests/test_strategy.py tests/test_backtest.py tests/test_stats.py tests/test_portfolio.py tests/test_kalman.py tests/test_viz.py -v
python -m pytest tests/test_research.py tests/test_ml_select.py tests/test_risk.py tests/test_live.py -v
```

All 59 tests use synthetic data — no network calls or API keys required.

## Design Principles

- **No lookahead bias** — signals shifted by 1 day; hedge ratios fit only on past data
- **Walk-forward methodology** — daily or quarterly refit windows, never peeking at future data
- **Frozen dataclasses** — immutable configuration and result objects throughout
- **Lazy imports** — optional dependencies (matplotlib, scikit-learn) imported only when needed
- **Realistic costs** — transaction fees, slippage, and short borrow costs modeled explicitly
- **FDR control** — Benjamini-Hochberg correction for multiple hypothesis testing across pairs
