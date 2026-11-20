"""Hedge ratios, spreads and z-scores.

Conventions used throughout the package
---------------------------------------
* Hedge ratios are estimated on **price levels**: ``y_t = alpha + beta * x_t + e_t``.
  ``beta`` is therefore a *share* ratio: one share of ``y`` is hedged with ``beta``
  shares of ``x``. Converting it to portfolio weights requires prices (see
  :func:`sarb.backtest.engine.pair_weights`).
* Any parameter series indexed at ``t`` is estimated from data up to and including
  ``t-1`` only, so it can be used to evaluate the spread at ``t`` without lookahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sarb.features.kalman import KalmanConfig, kalman_hedge_ratio


def fit_hedge_ratio(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    """OLS fit of ``y ~ alpha + beta * x``. Call this on the training window only."""
    x_ = sm.add_constant(np.asarray(x, dtype=float))
    model = sm.OLS(np.asarray(y, dtype=float), x_).fit()
    alpha, beta = float(model.params[0]), float(model.params[1])
    return alpha, beta


def compute_spread(y: pd.Series, x: pd.Series, alpha: float, beta: float) -> pd.Series:
    """Residual spread ``y - (alpha + beta * x)`` for fixed parameters."""
    return y - (alpha + beta * x)


def rolling_zscore(s: pd.Series, lookback: int) -> pd.Series:
    """Rolling z-score of ``s`` over a trailing window that *includes* ``t``."""
    mu = s.rolling(lookback).mean()
    sd = s.rolling(lookback).std(ddof=0)
    return (s - mu) / sd


def rolling_hedge_ratio(y: pd.Series, x: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    """Walk-forward OLS hedge ratio, vectorised.

    The values at index ``t`` are the OLS estimates fitted on the ``window``
    observations ending at ``t-1`` (i.e. ``[t-window, t)``), so they are known
    before the close of ``t``. The first ``window`` values are NaN.
    """
    roll = x.rolling(window)
    var_x = roll.var(ddof=0)
    cov_xy = roll.cov(y, ddof=0)
    beta = cov_xy / var_x.where(var_x > 0)
    alpha = y.rolling(window).mean() - beta * x.rolling(window).mean()
    return alpha.shift(1).rename("alpha"), beta.shift(1).rename("beta")


def spread_zscore_time_varying(
    y: pd.Series,
    x: pd.Series,
    alpha: pd.Series,
    beta: pd.Series,
    lookback: int,
) -> tuple[pd.Series, pd.Series]:
    """Spread and z-score when the hedge parameters change every day.

    For each ``t`` the z-score is computed exactly as if the spread history over
    the trailing ``lookback`` window were re-evaluated with *today's* parameters
    ``(alpha_t, beta_t)``::

        s_k = y_k - alpha_t - beta_t * x_k           for k in (t-lookback, t]
        z_t = (s_t - mean(s)) / std(s)

    Recomputing that history every day is O(n * lookback). Expanding the window
    moments gives the same answer in O(n)::

        mean(s) = mean(y) - alpha_t - beta_t * mean(x)
        var(s)  = var(y) + beta_t**2 * var(x) - 2 * beta_t * cov(x, y)

    Returns ``(spread, z)`` where ``spread_t = y_t - alpha_t - beta_t * x_t``.
    """
    roll_y = y.rolling(lookback)
    roll_x = x.rolling(lookback)
    mean_y, mean_x = roll_y.mean(), roll_x.mean()
    var_y, var_x = roll_y.var(ddof=0), roll_x.var(ddof=0)
    cov_xy = roll_x.cov(y, ddof=0)

    var_s = (var_y + beta**2 * var_x - 2.0 * beta * cov_xy).clip(lower=0.0)
    sd_s = np.sqrt(var_s)
    # alpha cancels in (s_t - mean(s)); it only matters for the spread level itself.
    dev = (y - mean_y) - beta * (x - mean_x)
    # Treat a numerically-zero window variance as undefined rather than dividing by ~0.
    scale = (var_y + beta**2 * var_x).abs()
    z = dev / sd_s.where(var_s > 1e-12 * scale)

    spread = y - alpha - beta * x
    return spread.rename("spread"), z.rename("z")


def compute_pair_signals(
    y: pd.Series,
    x: pd.Series,
    train_lookback: int,
    z_lookback: int,
    hedge_method: str = "ols",
    kalman_config: KalmanConfig | None = None,
) -> pd.DataFrame:
    """Single source of truth for pair signals (shared by backtests and live trading).

    Returns a frame with columns ``alpha, beta, spread, z``. Parameters at ``t``
    use data up to ``t-1`` only; ``z_t`` uses the price at ``t`` (the close at
    which the decision is made). The first ``train_lookback`` rows are NaN.

    hedge_method:
        ``"ols"``    rolling OLS over the previous ``train_lookback`` days.
        ``"kalman"`` online Kalman filter; the one-step-ahead (prior) state is used,
                     and the first ``train_lookback`` days are treated as burn-in.
    """
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    yy, xx = df["y"], df["x"]

    if hedge_method == "ols":
        alpha, beta = rolling_hedge_ratio(yy, xx, train_lookback)
    elif hedge_method == "kalman":
        kr = kalman_hedge_ratio(yy, xx, kalman_config or KalmanConfig())
        # Posterior at t-1 == prior for t under a random-walk state.
        alpha, beta = kr.alpha.shift(1), kr.beta.shift(1)
        burn_in = np.arange(len(df)) < train_lookback
        alpha = alpha.mask(burn_in)
        beta = beta.mask(burn_in)
    else:
        raise ValueError(f"Unknown hedge_method {hedge_method!r}; expected 'ols' or 'kalman'.")

    spread, z = spread_zscore_time_varying(yy, xx, alpha, beta, z_lookback)
    out = pd.DataFrame({"alpha": alpha, "beta": beta, "spread": spread, "z": z})
    out.loc[beta.isna(), ["spread", "z"]] = np.nan
    return out
