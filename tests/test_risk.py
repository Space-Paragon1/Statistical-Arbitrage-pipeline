from __future__ import annotations
import numpy as np
import pandas as pd

from sarb.risk.covariance import ledoit_wolf_shrinkage, correlation_aware_weights
from sarb.risk.limits import (
    RiskLimits,
    apply_position_limits,
    check_drawdown_breaker,
    compute_short_borrow_cost,
)


def test_ledoit_wolf_shrinkage_positive_definite():
    rng = np.random.default_rng(42)
    ret = pd.DataFrame(rng.normal(0, 0.01, (200, 4)), columns=["A", "B", "C", "D"])
    cov = ledoit_wolf_shrinkage(ret)
    assert cov.shape == (4, 4)
    eigenvalues = np.linalg.eigvalsh(cov.values)
    assert (eigenvalues > 0).all()


def test_ledoit_wolf_shrinkage_symmetric():
    rng = np.random.default_rng(42)
    ret = pd.DataFrame(rng.normal(0, 0.01, (100, 3)), columns=["X", "Y", "Z"])
    cov = ledoit_wolf_shrinkage(ret)
    np.testing.assert_array_almost_equal(cov.values, cov.values.T)


def test_correlation_aware_weights():
    rng = np.random.default_rng(42)
    ret = pd.DataFrame(rng.normal(0, 0.01, (200, 3)), columns=["P1", "P2", "P3"])
    w = correlation_aware_weights(ret, target_vol=0.008, max_weight=0.5)
    assert isinstance(w, pd.Series)
    assert len(w) == 3
    assert (w >= 0).all()
    assert (w <= 0.5 + 1e-10).all()
    assert abs(w.sum() - 1.0) < 1e-10


def test_apply_position_limits():
    limits = RiskLimits(max_position_size=0.3, max_gross_exposure=1.0)
    w = pd.Series([0.5, 0.4, 0.3], index=["A", "B", "C"])
    clipped = apply_position_limits(w, limits)
    assert (clipped.abs() <= 0.3 + 1e-10).all()
    assert clipped.abs().sum() <= 1.0 + 1e-10


def test_check_drawdown_breaker_triggers():
    # Equity drops 20% from peak
    eq = pd.Series([1.0, 1.1, 1.0, 0.88, 0.85, 0.9])
    limits = RiskLimits(max_drawdown_pct=0.15)
    assert check_drawdown_breaker(eq, limits) is True


def test_check_drawdown_breaker_no_trigger():
    eq = pd.Series([1.0, 1.01, 1.02, 1.01, 1.03])
    limits = RiskLimits(max_drawdown_pct=0.15)
    assert check_drawdown_breaker(eq, limits) is False


def test_compute_short_borrow_cost():
    short = pd.Series([10000.0, 10000.0, 10000.0])
    cost = compute_short_borrow_cost(short, borrow_cost_bps=100.0, ann_factor=252)
    # 100 bps annual = 1% annual => daily = 1%/252 => ~0.00397% per day
    expected_daily = 10000.0 * (100.0 / 1e4 / 252)
    np.testing.assert_almost_equal(cost.iloc[0], expected_daily)
