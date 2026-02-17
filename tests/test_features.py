from __future__ import annotations
import numpy as np
import pandas as pd

from sarb.features.spread import fit_hedge_ratio, compute_spread, rolling_zscore


def test_fit_hedge_ratio(synthetic_prices):
    alpha, beta = fit_hedge_ratio(synthetic_prices["Y"], synthetic_prices["X"])
    # Y ≈ 0.5 + 1.2*X, so beta should be close to 1.2
    assert abs(beta - 1.2) < 0.15
    assert abs(alpha - 0.5) < 5.0  # alpha absorbs level differences


def test_compute_spread(synthetic_prices):
    alpha, beta = fit_hedge_ratio(synthetic_prices["Y"], synthetic_prices["X"])
    spread = compute_spread(synthetic_prices["Y"], synthetic_prices["X"], alpha, beta)
    assert isinstance(spread, pd.Series)
    assert len(spread) == len(synthetic_prices)
    # Spread should be roughly stationary (low std relative to price levels)
    assert spread.std() < synthetic_prices["Y"].std()


def test_rolling_zscore(synthetic_prices):
    alpha, beta = fit_hedge_ratio(synthetic_prices["Y"], synthetic_prices["X"])
    spread = compute_spread(synthetic_prices["Y"], synthetic_prices["X"], alpha, beta)
    z = rolling_zscore(spread, 60)
    assert isinstance(z, pd.Series)
    assert len(z) == len(spread)
    # After warmup, z-scores should be roughly bounded
    z_valid = z.dropna()
    assert len(z_valid) > 0
    assert z_valid.abs().mean() < 5.0
