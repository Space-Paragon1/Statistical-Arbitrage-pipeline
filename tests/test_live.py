from __future__ import annotations
from datetime import datetime
import pandas as pd

from sarb.live.broker import Order
from sarb.live.paper_broker import PaperBroker
from sarb.live.signal import generate_live_signal, PairSignal
from sarb.live.runner import LiveConfig, run_live_step


def test_paper_broker_buy():
    broker = PaperBroker(initial_capital=100_000.0, fee_bps=1.0, slippage_bps=0.5)
    broker.set_current_prices({"AAPL": 150.0})

    fill = broker.submit_order(Order(ticker="AAPL", quantity=10.0, timestamp=datetime(2024, 1, 1)))
    assert fill is not None
    assert fill.quantity == 10.0
    assert fill.fill_price > 150.0  # slippage on buy

    assert broker.get_position("AAPL") == 10.0
    assert broker.get_account_value() < 100_000.0  # spent cash + fees


def test_paper_broker_sell():
    broker = PaperBroker(initial_capital=100_000.0, fee_bps=1.0, slippage_bps=0.5)
    broker.set_current_prices({"AAPL": 150.0})

    broker.submit_order(Order(ticker="AAPL", quantity=10.0, timestamp=datetime(2024, 1, 1)))
    broker.submit_order(Order(ticker="AAPL", quantity=-10.0, timestamp=datetime(2024, 1, 2)))

    assert broker.get_position("AAPL") == 0.0
    assert len(broker.get_fill_history()) == 2


def test_paper_broker_account_value():
    broker = PaperBroker(initial_capital=50_000.0)
    broker.set_current_prices({"X": 100.0})
    broker.submit_order(Order(ticker="X", quantity=100.0, timestamp=datetime(2024, 1, 1)))

    # Position value = 100 shares * $100 = $10,000
    # Cash decreased by ~$10,000 + fees
    acct = broker.get_account_value()
    # Should be close to initial capital (position value offsets cash spend)
    assert abs(acct - 50_000.0) < 100.0  # within $100 (fees + slippage)


def test_generate_live_signal(synthetic_prices):
    sig = generate_live_signal(
        synthetic_prices, y="Y", x="X",
        lookback_z=60, entry_z=2.0, exit_z=0.5, train_lookback=300,
    )
    assert isinstance(sig, PairSignal)
    assert sig.y_ticker == "Y"
    assert sig.x_ticker == "X"
    assert sig.position in (-1.0, 0.0, 1.0)
    assert isinstance(sig.alpha, float)
    assert isinstance(sig.beta, float)


def test_run_live_step(synthetic_prices):
    broker = PaperBroker(initial_capital=100_000.0)
    config = LiveConfig(
        pairs=[("Y", "X")],
        train_lookback=300,
        notional_per_pair=10_000.0,
    )

    # Use enough data for train_lookback + z_lookback
    px = synthetic_prices.iloc[:400]
    current = {t: float(px[t].iloc[-1]) for t in px.columns}
    broker.set_current_prices(current)

    signals = run_live_step(px, config, broker)
    assert isinstance(signals, list)
    assert len(signals) == 1
    assert isinstance(signals[0], PairSignal)
