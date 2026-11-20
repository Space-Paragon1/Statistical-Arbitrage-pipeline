"""Cointegration and mean-reversion diagnostics."""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

MIN_OBS = 50

# statsmodels >= 0.15 warns that adfuller will return a result object from 0.16 on;
# pin the tuple form where the switch exists so behaviour is identical across versions.
_ADF_KW = (
    {"result_object": False} if "result_object" in inspect.signature(adfuller).parameters else {}
)


@dataclass(frozen=True)
class CointResult:
    stat: float
    pvalue: float
    alpha: float
    beta: float


def engle_granger_test(y: pd.Series, x: pd.Series, maxlag: int | None = None) -> CointResult:
    """Two-step Engle-Granger cointegration test of ``y`` on ``x``.

    The residuals of ``y ~ alpha + beta * x`` are tested for a unit root using
    MacKinnon (1994/2010) *cointegration* critical values. Plain ADF critical
    values are invalid here: OLS chooses ``beta`` to make the residual look as
    stationary as possible, so ordinary ADF p-values are biased downwards
    (over-reject the null of no cointegration).
    """
    df = pd.concat([y, x], axis=1).dropna()
    if len(df) < MIN_OBS:
        return CointResult(float("nan"), float("nan"), float("nan"), float("nan"))
    yy, xx = df.iloc[:, 0].to_numpy(float), df.iloc[:, 1].to_numpy(float)
    autolag = None if maxlag is not None else "aic"
    stat, pvalue, _crit = coint(yy, xx, trend="c", maxlag=maxlag, autolag=autolag)
    params = sm.OLS(yy, sm.add_constant(xx)).fit().params
    return CointResult(float(stat), float(pvalue), float(params[0]), float(params[1]))


def engle_granger_adf_pvalue(spread: pd.Series, maxlag: int | None = None) -> float:
    """ADF p-value of a series, using ordinary (Dickey-Fuller) critical values.

    Use this for a spread whose parameters were **not** estimated on the same
    sample (e.g. out-of-sample residuals). For in-sample OLS residuals use
    :func:`engle_granger_test`, which applies the correct critical values.
    """
    s = spread.dropna().to_numpy(float)
    if len(s) < MIN_OBS:
        return float("nan")
    res = adfuller(s, maxlag=maxlag, regression="c", autolag="AIC", **_ADF_KW)
    return float(res[1])


def estimate_half_life(spread: pd.Series) -> float:
    """Half-life of mean reversion from an AR(1)/discretised OU fit.

    ``ds_t = a + b * s_{t-1} + e_t`` gives ``half_life = -ln 2 / ln(1 + b)``.
    (``-ln 2 / b`` is the small-``b`` approximation.) NaN if not mean-reverting.
    """
    s = spread.dropna()
    if len(s) < MIN_OBS:
        return float("nan")
    s_lag = s.shift(1).iloc[1:]
    ds = s.diff().iloc[1:]
    model = sm.OLS(ds.to_numpy(float), sm.add_constant(s_lag.to_numpy(float))).fit()
    b = float(model.params[1])
    if not -1.0 < b < 0.0:
        return float("nan")
    return float(-np.log(2.0) / np.log1p(b))
