"""Live signal generation.

Uses exactly the same signal path as the walk-forward backtest
(:func:`compute_pair_signals` + :func:`next_state`), so a strategy that is
backtested is the strategy that trades.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from sarb.features.spread import compute_pair_signals, fit_hedge_ratio
from sarb.strategy.pairs import PositionState, SignalRules, next_state


@dataclass(frozen=True)
class PairSignal:
    y_ticker: str
    x_ticker: str
    z_score: float
    position: float  # target spread position after today's close: +1, -1 or 0
    alpha: float
    beta: float  # share hedge ratio: short `beta` shares of x per share of y
    timestamp: pd.Timestamp
    bars_held: int = 0


def generate_live_signal(
    prices: pd.DataFrame,
    y: str,
    x: str,
    lookback_z: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    train_lookback: int = 504,
    hedge_method: str = "ols",
    current: PositionState | None = None,
    stop_z: float | None = None,
    max_holding: int | None = None,
) -> PairSignal:
    """Target position after observing the latest close.

    ``current`` is the strategy's state going into today (position and bars held).
    If omitted, the state is reconstructed by replaying the signal history, which
    is what a freshly started process should do.
    """
    rules = SignalRules(entry_z=entry_z, exit_z=exit_z, stop_z=stop_z, max_holding=max_holding)
    px = prices[[y, x]].dropna()
    ts = px.index[-1]
    if len(px) < train_lookback + 1:
        return PairSignal(y, x, 0.0, 0.0, 0.0, 0.0, ts)

    if hedge_method == "kalman":
        window = px  # the filter needs its whole history
    else:
        window = px.tail(train_lookback + lookback_z + 1) if current is not None else px
    sig = compute_pair_signals(window[y], window[x], train_lookback, lookback_z, hedge_method)
    z_hist = sig["z"]

    if current is None:
        st = PositionState()
        for zi in z_hist.iloc[:-1].to_numpy(dtype=float):
            st = next_state(st, zi, rules)
    else:
        st = current
    z_now = float(z_hist.iloc[-1])
    st = next_state(st, z_now, rules)

    # The hedge ratio to hold overnight: refit including today's close.
    if hedge_method == "kalman":
        from sarb.features.kalman import fit_hedge_ratio_kalman

        alpha, beta = fit_hedge_ratio_kalman(px[y], px[x])
    else:
        train = px.tail(train_lookback)
        alpha, beta = fit_hedge_ratio(train[y], train[x])

    return PairSignal(
        y_ticker=y,
        x_ticker=x,
        z_score=z_now if pd.notna(z_now) else 0.0,
        position=st.position,
        alpha=alpha,
        beta=beta,
        timestamp=ts,
        bars_held=st.bars_held,
    )
