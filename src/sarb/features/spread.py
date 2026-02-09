from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm

def fit_hedge_ratio(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    """Fit y ~ alpha + beta*x on TRAIN only."""
    x_ = sm.add_constant(x.values)
    model = sm.OLS(y.values, x_).fit()
    alpha, beta = float(model.params[0]), float(model.params[1])
    return alpha, beta

def compute_spread(y: pd.Series, x: pd.Series, alpha: float, beta: float) -> pd.Series:
    return y - (alpha + beta * x)

def rolling_zscore(s: pd.Series, lookback: int) -> pd.Series:
    mu = s.rolling(lookback).mean()
    sd = s.rolling(lookback).std(ddof=0)
    z = (s - mu) / sd
    return z
