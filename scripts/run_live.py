from __future__ import annotations
from sarb.data.ingest import load_yfinance_prices
from sarb.live.paper_broker import PaperBroker
from sarb.live.runner import LiveConfig, run_live_step


def main():
    pairs = [("KO", "PEP"), ("XOM", "CVX")]
    tickers = list({t for p in pairs for t in p})

    print("Loading price data...")
    prices = load_yfinance_prices(tickers, "2020-01-01", "2025-01-01", field="Adj Close")
    prices = prices.dropna()
    print(f"Loaded {len(prices)} days for {list(prices.columns)}")

    broker = PaperBroker(initial_capital=100_000.0, fee_bps=1.0, slippage_bps=0.5)
    config = LiveConfig(pairs=pairs, notional_per_pair=20_000.0)

    start_idx = config.train_lookback + config.lookback_z
    print(f"Starting paper trading from day {start_idx}...\n")

    for i in range(start_idx, len(prices)):
        daily_prices = prices.iloc[: i + 1]
        current = {t: float(daily_prices[t].iloc[-1]) for t in tickers}
        broker.set_current_prices(current)

        signals = run_live_step(daily_prices, config, broker)

        if i % 50 == 0 or i == len(prices) - 1:
            date = prices.index[i].strftime("%Y-%m-%d")
            acct = broker.get_account_value()
            pos = broker.get_all_positions()
            n_pos = len(pos)
            print(f"  {date} | Account: ${acct:,.2f} | Open positions: {n_pos}")

    final = broker.get_account_value()
    pnl = final - 100_000.0
    print(f"\n=== Paper Trading Summary ===")
    print(f"Final account value: ${final:,.2f}")
    print(f"Total P&L: ${pnl:,.2f} ({pnl / 100_000.0:.2%})")
    print(f"Total fills: {len(broker.get_fill_history())}")


if __name__ == "__main__":
    main()
