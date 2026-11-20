"""Vectorised, weight-based backtest engine.

Timing convention (daily bars, close-to-close)
----------------------------------------------
``weights.loc[t]`` is the portfolio *held over* bar ``t`` (from close ``t-1`` to
close ``t``) and must be decided with information available at close ``t-1``.
Trades that establish ``weights.loc[t]`` execute at close ``t-1``; their cost is
booked on bar ``t``.

Turnover is drift-adjusted: before rebalancing, yesterday's weights have drifted
with yesterday's returns, so holding a constant number of shares costs nothing
and only genuine trades are charged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def backtest_weights(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    cost_bps: float = 0.0,
    short_borrow_cost_bps: float = 0.0,
    ann_factor: int = TRADING_DAYS,
) -> pd.DataFrame:
    """Simulate a portfolio of asset weights (fractions of current equity).

    Parameters
    ----------
    weights:
        Target weights held over each bar, columns = assets. Already lagged.
    returns:
        Simple close-to-close asset returns, same index/columns as ``weights``.
    cost_bps:
        All-in one-way trading cost (fees + slippage) per unit of traded notional.
    short_borrow_cost_bps:
        Annualised borrow fee charged daily on short notional.

    Returns a frame with ``ret_gross, turnover, costs, ret_net, equity,
    gross_exposure, net_exposure``. Uninvested capital earns zero.
    """
    w = weights.fillna(0.0)
    r = returns.reindex(index=w.index, columns=w.columns)
    if (r.isna() & (w != 0.0)).any().any():
        bad = r.columns[(r.isna() & (w != 0.0)).any()].tolist()
        raise ValueError(f"Missing returns while holding a position in {bad}.")
    r = r.fillna(0.0)

    W = w.to_numpy(dtype=float)
    R = r.to_numpy(dtype=float)

    port_gross = (W * R).sum(axis=1)

    # Weights after bar t's returns, before rebalancing into bar t+1.
    drifted = W * (1.0 + R) / (1.0 + port_gross)[:, None]
    prev = np.vstack([np.zeros((1, W.shape[1])), drifted[:-1]])
    turnover = np.abs(W - prev).sum(axis=1)

    costs = turnover * cost_bps / 1e4
    if short_borrow_cost_bps:
        short_notional = np.clip(-W, 0.0, None).sum(axis=1)
        costs = costs + short_notional * short_borrow_cost_bps / 1e4 / ann_factor

    ret_net = port_gross - costs
    out = pd.DataFrame(
        {
            "ret_gross": port_gross,
            "turnover": turnover,
            "costs": costs,
            "ret_net": ret_net,
            "gross_exposure": np.abs(W).sum(axis=1),
            "net_exposure": W.sum(axis=1),
        },
        index=w.index,
    )
    out["equity"] = (1.0 + out["ret_net"]).cumprod()
    return out


def pair_weights(
    py: pd.Series,
    px: pd.Series,
    spread_pos: pd.Series,
    beta: float | pd.Series,
    leverage: float = 1.0,
) -> pd.DataFrame:
    """Convert a spread position into dollar weights for the two legs.

    ``beta`` is a share hedge ratio from a price-level regression, so one unit of
    spread is ``+1`` share of y and ``-beta`` shares of x. Its dollar weights are
    proportional to ``(P_y, -beta * P_x)``, normalised so gross exposure equals
    ``leverage``. Prices from the previous close are used because that is when
    the position is set; the resulting daily P&L is then exactly
    ``leverage * pos * (dP_y - beta * dP_x) / (P_y + |beta| * P_x)``,
    i.e. the P&L of the spread whose z-score generated the signal.
    """
    beta_s = (
        beta.reindex(spread_pos.index)
        if isinstance(beta, pd.Series)
        else pd.Series(float(beta), index=spread_pos.index)
    )
    py_lag = py.shift(1).fillna(py)
    px_lag = px.shift(1).fillna(px)
    dollar_y = py_lag
    dollar_x = -beta_s * px_lag
    gross = dollar_y.abs() + dollar_x.abs()
    k = (leverage * spread_pos / gross.where(gross > 0)).fillna(0.0)
    return pd.DataFrame({py.name: k * dollar_y, px.name: (k * dollar_x).fillna(0.0)})


def backtest_pairs(
    prices: pd.DataFrame,
    y: str,
    x: str,
    alpha: float,
    beta: float | pd.Series,
    spread_pos: pd.Series,
    fee_bps: float,
    slippage_bps: float,
    leverage: float = 1.0,
    short_borrow_cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Backtest one pair given an already-lagged spread position.

    ``alpha`` is accepted for API symmetry; it shifts the spread level but does not
    affect P&L. Output columns: ``w_y, w_x, turnover, ret_gross, costs, ret_net,
    equity, gross_exposure, net_exposure``.
    """
    del alpha  # spread level offset; irrelevant to P&L
    px = prices[[y, x]]
    pos = spread_pos.reindex(px.index).fillna(0.0)
    W = pair_weights(px[y], px[x], pos, beta, leverage)
    W.columns = [y, x]
    rets = px.pct_change().fillna(0.0)
    bt = backtest_weights(W, rets, fee_bps + slippage_bps, short_borrow_cost_bps)
    bt.insert(0, "w_x", W[x].to_numpy())
    bt.insert(0, "w_y", W[y].to_numpy())
    return bt
