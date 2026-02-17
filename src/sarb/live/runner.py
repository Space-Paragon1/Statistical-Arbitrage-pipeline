from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd

from sarb.live.broker import BaseBroker, Order
from sarb.live.signal import generate_live_signal, PairSignal
from sarb.risk.limits import RiskLimits


@dataclass(frozen=True)
class LiveConfig:
    pairs: list[tuple[str, str]]
    lookback_z: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5
    train_lookback: int = 504
    hedge_method: str = "ols"
    risk_limits: RiskLimits = field(default_factory=RiskLimits)
    notional_per_pair: float = 10_000.0


def run_live_step(
    prices: pd.DataFrame,
    config: LiveConfig,
    broker: BaseBroker,
) -> list[PairSignal]:
    """
    Single step: generate signals for all pairs, compute target positions,
    and submit orders to broker.
    """
    signals: list[PairSignal] = []
    account_value = broker.get_account_value()

    for y, x in config.pairs:
        sig = generate_live_signal(
            prices, y, x,
            lookback_z=config.lookback_z,
            entry_z=config.entry_z,
            exit_z=config.exit_z,
            train_lookback=config.train_lookback,
            hedge_method=config.hedge_method,
        )
        signals.append(sig)

        # Convert signal to target dollar positions
        notional = config.notional_per_pair
        target_y_shares = sig.position * notional / max(prices[y].iloc[-1], 1e-6)
        target_x_shares = sig.position * (-sig.beta) * notional / max(prices[x].iloc[-1], 1e-6)

        # Compute deltas from current positions
        current_y = broker.get_position(y)
        current_x = broker.get_position(x)

        delta_y = target_y_shares - current_y
        delta_x = target_x_shares - current_x

        ts = prices.index[-1]

        # Submit orders for non-trivial deltas
        if abs(delta_y) > 0.01:
            broker.submit_order(Order(ticker=y, quantity=delta_y, timestamp=ts))
        if abs(delta_x) > 0.01:
            broker.submit_order(Order(ticker=x, quantity=delta_x, timestamp=ts))

    return signals
