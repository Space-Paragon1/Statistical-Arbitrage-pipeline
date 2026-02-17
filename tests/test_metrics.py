from __future__ import annotations
import numpy as np
import pandas as pd

from sarb.metrics.performance import sharpe, max_drawdown, cagr


def test_sharpe_zero_returns():
    # All-zero returns => std=0, sharpe=0
    r = pd.Series([0.0] * 252)
    assert sharpe(r) == 0.0


def test_sharpe_positive():
    rng = np.random.default_rng(42)
    r = pd.Series(rng.normal(0.0005, 0.01, 252))
    sh = sharpe(r)
    assert isinstance(sh, float)
    assert np.isfinite(sh)


def test_max_drawdown_no_drawdown():
    eq = pd.Series([1.0, 1.01, 1.02, 1.03, 1.04])
    assert max_drawdown(eq) == 0.0


def test_max_drawdown_known():
    # Peak at 1.0, drops to 0.8 => dd = -20%
    eq = pd.Series([1.0, 0.9, 0.8, 0.85, 0.9])
    dd = max_drawdown(eq)
    assert abs(dd - (-0.2)) < 1e-10


def test_cagr_known():
    # Start at 1.0, end at 2.0 over 252 days => 1 year => CAGR = 100%
    eq = pd.Series(np.linspace(1.0, 2.0, 252))
    g = cagr(eq)
    assert abs(g - 1.0) < 0.01


def test_cagr_flat():
    eq = pd.Series([1.0] * 100)
    assert cagr(eq) == 0.0
