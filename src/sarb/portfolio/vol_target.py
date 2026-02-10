from __future__ import annotations
import numpy as np
import pandas as pd

def realized_vol(daily_returns: pd.Series) -> float:
    """
    Daily realized volatility estimate (std of daily returns).
    """
    r = daily_returns.dropna()
    if len(r) < 30:
        return float("nan")
    v = float(r.std(ddof=0))
    return v

def vol_target_scale(
    daily_returns: pd.Series,
    target_daily_vol: float,
    max_scale: float = 3.0,
    min_vol: float = 1e-6,
) -> float:
    """
    Scale factor = target_vol / estimated_vol
    - computed using past returns only (train+val)
    - capped to avoid insane leverage
    """
    v = realized_vol(daily_returns)
    if not np.isfinite(v) or v < min_vol:
        return 0.0
    s = target_daily_vol / v
    s = float(np.clip(s, 0.0, max_scale))
    return s
