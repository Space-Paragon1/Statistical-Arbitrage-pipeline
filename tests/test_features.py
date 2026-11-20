from __future__ import annotations

import pandas as pd
import pytest

from sarb.features.spread import (
    compute_pair_signals,
    compute_spread,
    fit_hedge_ratio,
    rolling_hedge_ratio,
    rolling_zscore,
    spread_zscore_time_varying,
)


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


def test_rolling_hedge_ratio_matches_naive_ols(synthetic_prices):
    y, x, T = synthetic_prices["Y"], synthetic_prices["X"], 120
    alpha, beta = rolling_hedge_ratio(y, x, T)
    assert beta.iloc[:T].isna().all()
    for i in (T, T + 1, 250, len(y) - 1):
        a_ref, b_ref = fit_hedge_ratio(y.iloc[i - T : i], x.iloc[i - T : i])  # data < t only
        assert beta.iloc[i] == pytest.approx(b_ref, rel=1e-8)
        assert alpha.iloc[i] == pytest.approx(a_ref, rel=1e-6, abs=1e-6)


def test_time_varying_zscore_matches_full_recompute(synthetic_prices):
    """The O(n) moment identity must equal re-evaluating the spread history with
    each day's parameters and z-scoring it (the original O(n^2) algorithm)."""
    y, x, T, L = synthetic_prices["Y"], synthetic_prices["X"], 120, 30
    alpha, beta = rolling_hedge_ratio(y, x, T)
    _, z = spread_zscore_time_varying(y, x, alpha, beta, L)
    for i in (T, T + 7, 300, len(y) - 1):
        s_hist = compute_spread(y.iloc[: i + 1], x.iloc[: i + 1], alpha.iloc[i], beta.iloc[i])
        z_ref = rolling_zscore(s_hist, L).iloc[-1]
        assert z.iloc[i] == pytest.approx(z_ref, rel=1e-7)


def test_compute_pair_signals_kalman_burn_in(synthetic_prices):
    sig = compute_pair_signals(synthetic_prices["Y"], synthetic_prices["X"], 100, 30, "kalman")
    assert sig["beta"].iloc[:100].isna().all()
    assert sig["z"].iloc[130:].notna().all()
    with pytest.raises(ValueError):
        compute_pair_signals(synthetic_prices["Y"], synthetic_prices["X"], 100, 30, "nope")
