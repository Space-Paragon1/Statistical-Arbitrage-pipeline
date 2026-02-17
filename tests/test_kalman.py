from __future__ import annotations
import numpy as np
import pandas as pd

from sarb.features.kalman import (
    kalman_hedge_ratio,
    fit_hedge_ratio_kalman,
    KalmanConfig,
    KalmanResult,
)


def test_kalman_hedge_ratio_convergence(synthetic_prices):
    """Final beta should converge near 1.2 on synthetic cointegrated data."""
    result = kalman_hedge_ratio(synthetic_prices["Y"], synthetic_prices["X"])
    assert isinstance(result, KalmanResult)
    assert len(result.beta) == len(synthetic_prices)
    assert len(result.alpha) == len(synthetic_prices)
    assert len(result.spread) == len(synthetic_prices)
    # Beta should converge toward 1.2 (true value)
    assert abs(result.beta.iloc[-1] - 1.2) < 0.2


def test_fit_hedge_ratio_kalman(synthetic_prices):
    alpha, beta = fit_hedge_ratio_kalman(synthetic_prices["Y"], synthetic_prices["X"])
    assert isinstance(alpha, float)
    assert isinstance(beta, float)
    assert np.isfinite(alpha)
    assert np.isfinite(beta)
    assert abs(beta - 1.2) < 0.2


def test_kalman_result_lengths(synthetic_prices):
    result = kalman_hedge_ratio(synthetic_prices["Y"], synthetic_prices["X"])
    n = len(synthetic_prices)
    assert len(result.alpha) == n
    assert len(result.beta) == n
    assert len(result.spread) == n
    assert len(result.measurement_error) == n


def test_kalman_custom_config(synthetic_prices):
    cfg = KalmanConfig(delta=1e-3, ve=1e-2, initial_beta=0.5, initial_alpha=0.0)
    result = kalman_hedge_ratio(synthetic_prices["Y"], synthetic_prices["X"], config=cfg)
    # Should still converge reasonably
    assert abs(result.beta.iloc[-1] - 1.2) < 0.3
