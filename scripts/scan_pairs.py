from __future__ import annotations
import pandas as pd

from sarb.data.ingest import load_yfinance_prices
from sarb.split.time_split import time_train_val_test_split
from sarb.research.select_pairs import scan_pairs
from sarb.features.spread import fit_hedge_ratio, compute_spread, rolling_zscore
from sarb.strategy.pairs import generate_spread_positions
from sarb.backtest.engine import backtest_pairs
from sarb.metrics.performance import sharpe, max_drawdown, cagr
from sarb.stats.bootstrap import bootstrap_mean_ci, bootstrap_sharpe_ci


def main():
    # Simple starter universe (you can expand later)
    tickers = [
        "KO","PEP","XOM","CVX","JPM","BAC","MSFT","AAPL",
        "SPY","IWM","QQQ","DIA","GLD","SLV","TLT","IEF"
    ]
    start, end = "2015-01-01", "2025-01-01"

    prices = load_yfinance_prices(tickers, start, end, field="Adj Close")
    prices = prices.dropna(axis=1, how="any")  # keep tickers with full history
    tickers = list(prices.columns)

    train_px, val_px, test_px = time_train_val_test_split(prices, train_frac=0.6, val_frac=0.2)
    train_idx, val_idx, test_idx = train_px.index, val_px.index, test_px.index

    # Selection (NO test usage here)
    selected = scan_pairs(
        prices=prices,
        tickers=tickers,
        train_idx=train_idx,
        val_idx=val_idx,
        lookback_z=60,
        entry_z=2.0,
        exit_z=0.5,
        fee_bps=1.0,
        slippage_bps=0.5,
        leverage=1.0,
        corr_threshold=0.6,
        max_pairs=250,
        fdr_q=0.10,
        top_k=5,
    )

    if not selected:
        print("No pairs selected. Try lowering corr_threshold or expanding tickers.")
        return

    print("\n=== Selected Pairs (based on VALIDATION only) ===")
    for r in selected:
        print(
            f"{r.y}/{r.x} | valSharpe={r.val_sharpe:.2f} | ADFp={r.adf_p:.4f} | HL={r.half_life:.1f} | corr={r.corr:.2f}"
        )

    # Evaluate each selected pair on TEST only (reporting)
    print("\n=== Out-of-sample TEST Results ===")
    for r in selected:
        # Fit on TRAIN only (you can also fit on TRAIN+VAL for a different protocol; keep train-only for purity)
        alpha, beta = fit_hedge_ratio(train_px[r.y], train_px[r.x])

        # Build signals on TRAIN+VAL+TEST but execute lookahead-safe (shifted positions)
        spread = compute_spread(prices[r.y], prices[r.x], alpha, beta)
        z = rolling_zscore(spread, 60)
        pos = generate_spread_positions(z, 2.0, 0.5)

        bt = backtest_pairs(
            prices=prices[[r.y, r.x]].dropna(),
            y=r.y,
            x=r.x,
            alpha=alpha,
            beta=beta,
            spread_pos=pos,
            fee_bps=1.0,
            slippage_bps=0.5,
            leverage=1.0,
        )

        bt_test = bt.loc[test_idx.intersection(bt.index)]
        daily = bt_test["ret_net"]
        eq = bt_test["equity"]

        sh = sharpe(daily)
        mdd = max_drawdown(eq)
        g = cagr(eq)

        mean_ci = bootstrap_mean_ci(daily, n_boot=2000)
        sh_ci = bootstrap_sharpe_ci(daily, n_boot=2000)

        print(f"\nPair {r.y}/{r.x}")
        print(f"Sharpe={sh:.2f} | MaxDD={mdd:.2%} | CAGR={g:.2%}")
        print(f"Mean CI: [{mean_ci['ci_low']:.6f}, {mean_ci['ci_high']:.6f}]")
        print(f"Sharpe CI: [{sh_ci['ci_low']:.2f}, {sh_ci['ci_high']:.2f}]")

if __name__ == "__main__":
    main()
