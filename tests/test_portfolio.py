from __future__ import annotations
import numpy as np
import pandas as pd

from sarb.portfolio.vol_target import realized_vol, vol_target_scale


def test_realized_vol():
    rng = np.random.default_rng(42)
    r = pd.Series(rng.normal(0, 0.01, 252))
    v = realized_vol(r)
    assert np.isfinite(v)
    assert abs(v - 0.01) < 0.005  # should be close to true vol


def test_realized_vol_short_series():
    r = pd.Series([0.01] * 10)
    v = realized_vol(r)
    assert np.isnan(v)  # fewer than 30 observations


def test_vol_target_scale():
    rng = np.random.default_rng(42)
    r = pd.Series(rng.normal(0, 0.01, 252))
    scale = vol_target_scale(r, target_daily_vol=0.008, max_scale=3.0)
    assert isinstance(scale, float)
    assert 0 < scale <= 3.0
    # With vol ≈ 0.01, target 0.008 => scale ≈ 0.8
    assert abs(scale - 0.8) < 0.3


def test_vol_target_scale_capped():
    rng = np.random.default_rng(42)
    # Very low vol series => scale would be huge, should be capped
    r = pd.Series(rng.normal(0, 0.001, 252))
    scale = vol_target_scale(r, target_daily_vol=0.01, max_scale=3.0)
    assert scale == 3.0
