from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

def engle_granger_adf_pvalue(spread: pd.Series, maxlag: int | None = None) -> float:
    """
    Engle–Granger style check: if spread is stationary, pair is (often) cointegrated.
    We run ADF on spread (residual). Lower p-value => stronger evidence of stationarity.
    """
    s = spread.dropna().values
    if len(s) < 50:
        return float("nan")
    res = adfuller(s, maxlag=maxlag, regression="c", autolag="AIC")
    return float(res[1])  # p-value

def estimate_half_life(spread: pd.Series) -> float:
    """
    Estimate half-life of mean reversion using AR(1) approximation:
      Δs_t = a + b*s_{t-1} + e_t
    half-life = -ln(2) / b  (when b < 0)
    Returns NaN if not mean reverting.
    """
    s = spread.dropna()
    if len(s) < 50:
        return float("nan")

    s_lag = s.shift(1).dropna()
    ds = (s - s.shift(1)).dropna()
    # align
    ds = ds.loc[s_lag.index]

    X = sm.add_constant(s_lag.values)
    model = sm.OLS(ds.values, X).fit()
    b = float(model.params[1])

    if b >= 0:
        return float("nan")

    hl = -np.log(2) / b
    return float(hl)
