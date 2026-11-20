"""Inference on Sharpe ratios under non-normality and multiple testing.

References
----------
* Bailey, D. & López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier."
  Journal of Risk 15(2). -- Probabilistic Sharpe Ratio (PSR).
* Bailey, D. & López de Prado, M. (2014). "The Deflated Sharpe Ratio: Correcting
  for Selection Bias, Backtest Overfitting and Non-Normality."
  Journal of Portfolio Management 40(5). -- Deflated Sharpe Ratio (DSR).

All Sharpe ratios in this module are **per-period** (not annualised) unless the
argument name says otherwise; the formulas are stated in per-period units.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def _moments(r: np.ndarray) -> tuple[float, float, float, int]:
    n = len(r)
    sd = r.std(ddof=1)
    sr = r.mean() / sd if sd > 0 else 0.0
    skew = float(stats.skew(r, bias=False)) if n > 2 else 0.0
    kurt = float(stats.kurtosis(r, fisher=False, bias=False)) if n > 3 else 3.0
    return float(sr), skew, kurt, n


def sharpe_standard_error(sr: float, n: int, skew: float = 0.0, kurtosis: float = 3.0) -> float:
    """Std. error of a per-period Sharpe estimate (Mertens 2002 / Lo 2002 form)."""
    var = (1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr**2) / (n - 1)
    return float(np.sqrt(max(var, 1e-18)))


def probabilistic_sharpe_ratio(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """P(true Sharpe > ``sr_benchmark``), accounting for sample length, skew and kurtosis.

    ``sr_benchmark`` is per-period (e.g. ``0.5 / sqrt(252)`` for an annual 0.5).
    """
    r = returns.dropna().to_numpy(float)
    if len(r) < 3 or r.std(ddof=1) == 0:
        return float("nan")
    sr, skew, kurt, n = _moments(r)
    se = sharpe_standard_error(sr, n, skew, kurt)
    return float(stats.norm.cdf((sr - sr_benchmark) / se))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum of ``n_trials`` per-period Sharpe estimates under the null
    (true SR = 0), given the cross-sectional variance of those estimates."""
    if n_trials <= 1 or sr_variance <= 0:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sr_variance) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int,
    sr_variance: float,
) -> float:
    """PSR evaluated against the Sharpe ratio you would expect from the best of
    ``n_trials`` skill-less strategies. A DSR above 0.95 means the selected
    strategy's Sharpe is unlikely to be a multiple-testing artefact."""
    sr0 = expected_max_sharpe(n_trials, sr_variance)
    return probabilistic_sharpe_ratio(returns, sr_benchmark=sr0)


def per_period_sharpe(returns: pd.Series) -> float:
    r = returns.dropna().to_numpy(float)
    if len(r) < 2:
        return float("nan")
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else 0.0
