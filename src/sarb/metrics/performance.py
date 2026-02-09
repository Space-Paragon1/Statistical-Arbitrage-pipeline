from __future__ import annotations
import numpy as np
import pandas as pd

def sharpe(daily_returns: pd.Series, ann_factor: int = 252) -> float:
    r = daily_returns.dropna()
    if r.std(ddof=0) == 0:
        return 0.0
    return float((r.mean() / r.std(ddof=0)) * np.sqrt(ann_factor))

def max_drawdown(equity: pd.Series) -> float:
    eq = equity.dropna()
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float(dd.min()) if len(dd) else 0.0

def cagr(equity: pd.Series, ann_factor: int = 252) -> float:
    eq = equity.dropna()
    if len(eq) < 2:
        return 0.0
    total = eq.iloc[-1] / eq.iloc[0]
    years = len(eq) / ann_factor
    return float(total ** (1 / years) - 1) if years > 0 else 0.0
