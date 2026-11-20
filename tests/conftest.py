from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import pytest


def make_cointegrated_universe(
    n: int = 1000, seed: int = 7, phi: float = 0.9, noise: float = 0.6
) -> pd.DataFrame:
    """Two cointegrated pairs (A/B, C/D) plus an unrelated random walk E."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-01", periods=n)

    def ou(scale: float) -> np.ndarray:
        e = np.zeros(n)
        for i in range(1, n):
            e[i] = phi * e[i - 1] + rng.normal(0, scale)
        return e

    a = 80.0 + np.cumsum(rng.normal(0.02, 0.8, n))
    c = 40.0 + np.cumsum(rng.normal(0.01, 0.5, n))
    return pd.DataFrame(
        {
            "A": a,
            "B": 5.0 + 0.6 * a + ou(noise),
            "C": c,
            "D": 2.0 + 1.5 * c + ou(noise),
            "E": 60.0 + np.cumsum(rng.normal(0.0, 0.7, n)),
        },
        index=dates,
    ).clip(lower=1.0)


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """Two cointegrated price series (Y ≈ 0.5 + 1.2*X) plus one unrelated, 500 days."""
    rng = np.random.default_rng(42)
    n = 500
    dates = pd.bdate_range("2020-01-01", periods=n)

    x = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    ou = np.zeros(n)
    for i in range(1, n):
        ou[i] = 0.8 * ou[i - 1] + rng.normal(0, 0.3)
    y = 0.5 + 1.2 * x + ou
    z = 50.0 + np.cumsum(rng.normal(0, 0.4, n))
    return pd.DataFrame({"Y": y, "X": x, "Z": z}, index=dates)


@pytest.fixture
def synthetic_returns(synthetic_prices: pd.DataFrame) -> pd.DataFrame:
    return synthetic_prices.pct_change().dropna()


@pytest.fixture
def universe() -> pd.DataFrame:
    return make_cointegrated_universe()
