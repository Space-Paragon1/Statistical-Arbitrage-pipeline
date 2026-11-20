"""Walk-forward single-pair backtest with a daily hedge-ratio refit."""

from __future__ import annotations

import pandas as pd

from sarb.backtest.engine import backtest_weights, pair_weights
from sarb.features.kalman import KalmanConfig
from sarb.features.spread import compute_pair_signals
from sarb.strategy.pairs import SignalRules, spread_states


def walkforward_pairs_backtest(
    prices: pd.DataFrame,
    y: str,
    x: str,
    train_lookback: int,
    z_lookback: int,
    entry_z: float,
    exit_z: float,
    fee_bps: float,
    slippage_bps: float,
    leverage: float = 1.0,
    hedge_method: str = "ols",
    stop_z: float | None = None,
    max_holding: int | None = None,
    short_borrow_cost_bps: float = 0.0,
    kalman_config: KalmanConfig | None = None,
) -> pd.DataFrame:
    """Walk-forward backtest: refit the hedge ratio every day on past data only.

    For each bar ``t``:

    1. ``alpha_t, beta_t`` are fitted on ``[t - train_lookback, t)`` (OLS) or are
       the Kalman prior given data through ``t-1``.
    2. ``z_t`` is the z-score of today's spread over the trailing ``z_lookback``
       window, evaluated with today's parameters.
    3. A single persistent state machine turns ``z_t`` into the position decided
       at close ``t``, which is *held over* ``t+1`` together with the hedge
       ratio known at close ``t``.

    Everything is vectorised except the O(n) state machine, so a 10-year daily
    backtest runs in milliseconds.
    """
    px = prices[[y, x]].dropna()
    sig = compute_pair_signals(
        px[y], px[x], train_lookback, z_lookback, hedge_method, kalman_config
    )
    rules = SignalRules(entry_z=entry_z, exit_z=exit_z, stop_z=stop_z, max_holding=max_holding)
    state = spread_states(sig["z"], rules)

    pos = state.shift(1).fillna(0.0)
    # sig["beta"] at t is fitted on data <= t-1, i.e. it is known at close t-1 and
    # is the freshest hedge ratio available when the position for bar t is set.
    W = pair_weights(px[y], px[x], pos, sig["beta"], leverage)
    W.columns = [y, x]
    rets = px.pct_change().fillna(0.0)
    bt = backtest_weights(W, rets, fee_bps + slippage_bps, short_borrow_cost_bps)

    out = pd.DataFrame(
        {
            "alpha": sig["alpha"],
            "beta": sig["beta"],
            "spread": sig["spread"],
            "z": sig["z"],
            "pos": pos,
            "w_y": W[y],
            "w_x": W[x],
        },
        index=px.index,
    )
    return pd.concat([out, bt], axis=1)
