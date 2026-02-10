from __future__ import annotations
import pandas as pd

from sarb.config import PairConfig
from sarb.data.ingest import load_yfinance_prices
from sarb.split.time_split import time_train_test_split
from sarb.features.spread import fit_hedge_ratio, compute_spread, rolling_zscore
from sarb.strategy.pairs import generate_spread_positions
from sarb.backtest.engine import backtest_pairs
from sarb.metrics.performance import sharpe, max_drawdown, cagr
from sarb.stats.bootstrap import bootstrap_mean_ci, bootstrap_sharpe_ci
from sarb.stats.cointegration import engle_granger_adf_pvalue, estimate_half_life
from sarb.backtest.walkforward import walkforward_pairs_backtest

def main():
    cfg = PairConfig()

    prices = load_yfinance_prices([cfg.y_ticker, cfg.x_ticker], cfg.start, cfg.end, field=cfg.price_field)
    prices = prices[[cfg.y_ticker, cfg.x_ticker]].dropna()

    train_px, test_px = time_train_test_split(prices, cfg.train_frac)

    # Fit hedge ratio on TRAIN only
    y_tr = train_px[cfg.y_ticker]
    x_tr = train_px[cfg.x_ticker]
    alpha, beta = fit_hedge_ratio(y_tr, x_tr)

    # Build spread/z-score on FULL data using fixed alpha/beta (no refit on test)
    y_full = prices[cfg.y_ticker]
    x_full = prices[cfg.x_ticker]
    spread = compute_spread(y_full, x_full, alpha, beta)
    z = rolling_zscore(spread, cfg.lookback_z)

    # Research diagnostics (on TRAIN spread to avoid test leakage)
    spread_train = spread.loc[train_px.index]
    adf_p = engle_granger_adf_pvalue(spread_train)
    hl = estimate_half_life(spread_train)

    print("=== Research Diagnostics (Train Window) ===")
    print(f"ADF p-value (spread stationarity): {adf_p:.6f}")
    print(f"Estimated half-life (days): {hl:.2f}")


    # Signals / positions
    pos = generate_spread_positions(z, cfg.entry_z, cfg.exit_z)

    # Backtest on full period, then evaluate on test only
    bt = walkforward_pairs_backtest(
        prices=prices,
        y=cfg.y_ticker,
        x=cfg.x_ticker,
        train_lookback=504,      # ~2 years of trading days
        z_lookback=cfg.lookback_z,
        entry_z=cfg.entry_z,
        exit_z=cfg.exit_z,
        fee_bps=cfg.fee_bps,
        slippage_bps=cfg.slippage_bps,
        leverage=cfg.leverage,
    )


    # Slice test window
    test_idx = test_px.index
    bt_test = bt.loc[test_idx]

    r = bt_test["ret_net"]
    eq = bt_test["equity"]

    print("=== Pairs Trading (Test Window) ===")
    print(f"Pair: {cfg.y_ticker}/{cfg.x_ticker}")
    print(f"alpha={alpha:.6f}, beta={beta:.6f}")
    print(f"Sharpe: {sharpe(r):.3f}")
    print(f"Max Drawdown: {max_drawdown(eq):.3%}")
    print(f"CAGR: {cagr(eq):.3%}")
    print(f"Avg daily net ret: {r.mean():.6f}")

    mean_ci = bootstrap_mean_ci(r, n_boot=3000)
    sh_ci = bootstrap_sharpe_ci(r, n_boot=3000)

    print("\n=== Bootstrap (Test Window) ===")
    print(f"Mean CI: [{mean_ci['ci_low']:.6f}, {mean_ci['ci_high']:.6f}] (mean={mean_ci['mean']:.6f})")
    print(f"Sharpe CI: [{sh_ci['ci_low']:.3f}, {sh_ci['ci_high']:.3f}] (sharpe={sh_ci['sharpe']:.3f})")

if __name__ == "__main__":
    main()
