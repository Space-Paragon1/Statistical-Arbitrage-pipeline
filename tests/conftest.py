from __future__ import annotations
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """Two cointegrated price series (Y ≈ 0.5 + 1.2*X) plus one unrelated, 500 days."""
    rng = np.random.default_rng(42)
    n = 500
    dates = pd.bdate_range("2020-01-01", periods=n)

    # X follows a random walk
    x = 100.0 + np.cumsum(rng.normal(0, 0.5, n))

    # Y = 0.5 + 1.2*X + stationary OU noise
    ou = np.zeros(n)
    for i in range(1, n):
        ou[i] = 0.8 * ou[i - 1] + rng.normal(0, 0.3)
    y = 0.5 + 1.2 * x + ou

    # Z is unrelated
    z = 50.0 + np.cumsum(rng.normal(0, 0.4, n))

    return pd.DataFrame({"Y": y, "X": x, "Z": z}, index=dates)


@pytest.fixture
def synthetic_returns(synthetic_prices: pd.DataFrame) -> pd.DataFrame:
    return synthetic_prices.pct_change().dropna()
