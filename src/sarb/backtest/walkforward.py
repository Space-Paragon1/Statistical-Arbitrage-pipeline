from __future__ import annotations
import pandas as pd

from sarb.features.spread import fit_hedge_ratio, compute_spread, rolling_zscore
from sarb.strategy.pairs import generate_spread_positions
from sarb.backtest.engine import backtest_pairs

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
) -> pd.DataFrame:
    """
    Walk-forward:
    For each day t after you have train_lookback days, refit alpha/beta on the prior window,
    compute spread/z, generate today's position from yesterday's z (no lookahead),
    then apply returns with costs.
    """
    px = prices[[y, x]].dropna().copy()

    # We'll build positions day-by-day to ensure time correctness
    pos = pd.Series(index=px.index, data=0.0)
    spread_all = pd.Series(index=px.index, data=float("nan"))
    z_all = pd.Series(index=px.index, data=float("nan"))

    for i in range(train_lookback, len(px)):
        train_slice = px.iloc[i - train_lookback : i]
        alpha, beta = fit_hedge_ratio(train_slice[y], train_slice[x])

        # compute spread up to i (inclusive) using current alpha/beta
        s = compute_spread(px[y].iloc[: i + 1], px[x].iloc[: i + 1], alpha, beta)
        z = rolling_zscore(s, z_lookback)

        spread_all.iloc[i] = s.iloc[-1]
        z_all.iloc[i] = z.iloc[-1]

        # generate position series up to i and take last (shifted inside generator)
        p = generate_spread_positions(z, entry_z, exit_z)
        pos.iloc[i] = p.iloc[-1]

    # Now run one backtest pass using the final positions.
    # For costs/weights we need a single beta; but in walk-forward beta changes over time.
    # So we approximate with beta=1 in engine? No.
    # Better: rebuild weights per day using rolling beta series.
    # We'll implement a walk-forward engine directly here.

    ret = px.pct_change().fillna(0.0)

    # compute rolling beta series again (store daily beta)
    beta_series = pd.Series(index=px.index, data=float("nan"))
    alpha_series = pd.Series(index=px.index, data=float("nan"))
    for i in range(train_lookback, len(px)):
        train_slice = px.iloc[i - train_lookback : i]
        a, b = fit_hedge_ratio(train_slice[y], train_slice[x])
        alpha_series.iloc[i] = a
        beta_series.iloc[i] = b
    beta_series = beta_series.ffill().fillna(0.0)

    w_y = pos * 1.0
    w_x = pos * (-beta_series)

    gross = (w_y.abs() + w_x.abs()).replace(0.0, pd.NA)
    scale = (leverage / gross).fillna(0.0)
    w_y = w_y * scale
    w_x = w_x * scale

    port_ret_gross = w_y * ret[y] + w_x * ret[x]

    dw_y = w_y.diff().abs().fillna(w_y.abs())
    dw_x = w_x.diff().abs().fillna(w_x.abs())
    turnover = dw_y + dw_x

    cost_rate = (fee_bps + slippage_bps) / 1e4
    costs = turnover * cost_rate
    port_ret_net = port_ret_gross - costs

    out = pd.DataFrame(
        {
            "alpha": alpha_series,
            "beta": beta_series,
            "z": z_all,
            "pos": pos,
            "w_y": w_y,
            "w_x": w_x,
            "turnover": turnover,
            "ret_gross": port_ret_gross,
            "costs": costs,
            "ret_net": port_ret_net,
        },
        index=px.index,
    )
    out["equity"] = (1.0 + out["ret_net"]).cumprod()
    return out
