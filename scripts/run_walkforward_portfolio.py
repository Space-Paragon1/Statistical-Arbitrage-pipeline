from __future__ import annotations

from sarb.data.ingest import load_yfinance_prices
from sarb.split.rebalance import rolling_windows_by_quarter
from sarb.research.walkforward_portfolio import WFConfig, walkforward_quarterly_portfolio
from sarb.metrics.performance import sharpe, max_drawdown, cagr


def summarize(label: str, port):
    r = port["ret_net"]
    eq = port["equity"]
    print(f"\n=== {label} ===")
    print(f"Sharpe: {sharpe(r):.2f}")
    print(f"Max Drawdown: {max_drawdown(eq):.2%}")
    print(f"CAGR: {cagr(eq):.2%}")


def main():
    tickers = [
        "KO","PEP","XOM","CVX","JPM","BAC","MSFT","AAPL",
        "SPY","IWM","QQQ","DIA","GLD","SLV","TLT","IEF"
    ]

    prices = load_yfinance_prices(tickers, "2015-01-01", "2025-01-01", field="Adj Close")
    prices = prices.dropna(axis=1, how="any")
    tickers = list(prices.columns)

    windows = rolling_windows_by_quarter(prices, train_days=504, val_days=126)

    # -------------------------
    # RUN 1: Equal weight (baseline)
    # -------------------------
    cfg_eq = WFConfig(
        top_k=5,
        corr_threshold=0.6,
        use_vol_targeting=False,   # OFF
    )
    port_eq, meta_eq = walkforward_quarterly_portfolio(prices, tickers, windows, cfg_eq)

    # -------------------------
    # RUN 2: Vol targeted (equal risk)
    # -------------------------
    cfg_vt = WFConfig(
        top_k=5,
        corr_threshold=0.6,
        use_vol_targeting=True,    # ON
        target_daily_vol=0.008,
        max_pair_scale=3.0,
    )
    port_vt, meta_vt = walkforward_quarterly_portfolio(prices, tickers, windows, cfg_vt)

    # -------------------------
    # Print metrics for both
    # -------------------------
    summarize("Equal Weight (no vol targeting)", port_eq)
    summarize("Vol Targeted (equal risk)", port_vt)

    print("\n=== Vol Target: Quarterly Selection Log (first 10) ===")
    print(meta_vt.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
