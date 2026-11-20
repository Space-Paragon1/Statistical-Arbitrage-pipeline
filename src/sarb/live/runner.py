"""Daily live/paper execution loop."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from sarb.live.broker import BaseBroker, Order
from sarb.live.signal import PairSignal, generate_live_signal
from sarb.risk.limits import RiskLimits
from sarb.strategy.pairs import PositionState


@dataclass(frozen=True)
class LiveConfig:
    pairs: list[tuple[str, str]]
    lookback_z: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float | None = None
    max_holding: int | None = None
    train_lookback: int = 504
    hedge_method: str = "ols"
    risk_limits: RiskLimits = field(default_factory=RiskLimits)
    notional_per_pair: float = 10_000.0  # gross notional (|long| + |short|) per open pair
    min_trade_shares: float = 0.01


@dataclass
class LiveState:
    """Per-pair strategy state carried between steps."""

    pairs: dict[tuple[str, str], PositionState] = field(default_factory=dict)


def target_shares(
    position: float, beta: float, price_y: float, price_x: float, gross_notional: float
) -> tuple[float, float]:
    """Shares of (y, x) for a spread position with the given gross notional.

    One unit of spread is +1 share of y and -beta shares of x, so the share ratio
    must be ``beta`` (not the dollar ratio) for the book to track the spread.
    """
    unit_gross = abs(price_y) + abs(beta) * abs(price_x)
    if position == 0.0 or unit_gross <= 0:
        return 0.0, 0.0
    n_y = position * gross_notional / unit_gross
    return n_y, -beta * n_y


def run_live_step(
    prices: pd.DataFrame,
    config: LiveConfig,
    broker: BaseBroker,
    state: LiveState | None = None,
) -> list[PairSignal]:
    """Generate signals for all pairs from the latest close and rebalance the broker.

    Pass the same ``state`` object on every call to carry each pair's position
    state (and bars held) forward; without it, state is rebuilt from history.
    Target shares are aggregated across pairs before ordering, so a ticker that
    appears in two pairs is netted rather than traded twice.
    """
    signals: list[PairSignal] = []
    targets: dict[str, float] = {}
    ts = prices.index[-1]

    for y, x in config.pairs:
        current = state.pairs.get((y, x)) if state is not None else None
        sig = generate_live_signal(
            prices,
            y,
            x,
            lookback_z=config.lookback_z,
            entry_z=config.entry_z,
            exit_z=config.exit_z,
            train_lookback=config.train_lookback,
            hedge_method=config.hedge_method,
            current=current,
            stop_z=config.stop_z,
            max_holding=config.max_holding,
        )
        signals.append(sig)
        if state is not None:
            state.pairs[(y, x)] = PositionState(sig.position, sig.bars_held)

        n_y, n_x = target_shares(
            sig.position,
            sig.beta,
            float(prices[y].iloc[-1]),
            float(prices[x].iloc[-1]),
            config.notional_per_pair,
        )
        targets[y] = targets.get(y, 0.0) + n_y
        targets[x] = targets.get(x, 0.0) + n_x

    for tkr in set(targets) | set(broker.get_all_positions()):
        delta = targets.get(tkr, 0.0) - broker.get_position(tkr)
        if abs(delta) > config.min_trade_shares:
            broker.submit_order(Order(ticker=tkr, quantity=delta, timestamp=ts))

    return signals
