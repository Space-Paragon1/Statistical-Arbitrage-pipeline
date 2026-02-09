from __future__ import annotations
import numpy as np
import pandas as pd

def backtest_pairs(
    prices: pd.DataFrame,
    y: str,
    x: str,
    alpha: float,
    beta: float,
    spread_pos: pd.Series,
    fee_bps: float,
    slippage_bps: float,
    leverage: float = 1.0,
) -> pd.DataFrame:
    """
    prices: DataFrame with columns [y, x] of aligned prices
    spread_pos: +1 long spread, -1 short spread, 0 flat (already shifted)
    Implements self-financing legs:
      long spread: +1*y  and  -beta*x
      short spread: -1*y and +beta*x
    Uses simple returns per leg and charges costs on position changes (turnover).
    """
    px = prices[[y, x]].copy()
    ret = px.pct_change().fillna(0.0)

    # Leg weights for 1 unit of spread
    w_y = spread_pos * 1.0
    w_x = spread_pos * (-beta)

    # scale gross exposure to leverage
    gross = (w_y.abs() + w_x.abs()).replace(0.0, np.nan)
    scale = (leverage / gross).fillna(0.0)
    w_y = w_y * scale
    w_x = w_x * scale

    port_ret_gross = w_y * ret[y] + w_x * ret[x]

    # Transaction costs based on changes in weights (turnover)
    dw_y = w_y.diff().abs().fillna(w_y.abs())
    dw_x = w_x.diff().abs().fillna(w_x.abs())
    turnover = dw_y + dw_x

    cost_rate = (fee_bps + slippage_bps) / 1e4  # bps -> fraction
    costs = turnover * cost_rate

    port_ret_net = port_ret_gross - costs

    out = pd.DataFrame(
        {
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
