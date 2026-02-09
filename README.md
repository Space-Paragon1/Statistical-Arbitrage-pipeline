# Statistical Arbitrage Research Pipeline (Pairs Trading)

A research-oriented statistical arbitrage pipeline focused on correctness and rigor:
- Data ingestion (yfinance or CSV)
- Time-aware train/test split
- Cointegration-inspired pairs trading via Engle–Granger hedge ratio (train-only fit)
- Z-score mean reversion signals with lookahead-safe execution
- Backtesting with transaction costs + slippage
- Statistical validation (Sharpe, drawdown, bootstrap confidence intervals)

## Strategy Overview
We model a pair (Y, X) with a linear hedge ratio on the training window:
Y_t ≈ α + β X_t
Spread: s_t = Y_t - (α + β X_t)
Z-score: z_t = (s_t - rolling_mean) / rolling_std

Rules:
- Enter long spread when z < -entry_z
- Enter short spread when z > +entry_z
- Exit when |z| < exit_z
Signals are shifted by 1 day to avoid lookahead bias.

## Project Structure
See `src/sarb/` modules for ingestion, features, split, strategy, backtest, metrics, and stats.

## Quickstart
```bash
pip install -e .
python scripts/run_pairs.py
