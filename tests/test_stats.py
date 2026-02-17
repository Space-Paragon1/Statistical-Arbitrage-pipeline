from __future__ import annotations
import numpy as np
import pandas as pd

from sarb.stats.cointegration import engle_granger_adf_pvalue, estimate_half_life
from sarb.stats.bootstrap import bootstrap_mean_ci, bootstrap_sharpe_ci
from sarb.stats.multiple_testing import benjamini_hochberg


def test_adf_stationary(synthetic_prices):
    """Spread of cointegrated pair should have low ADF p-value."""
    from sarb.features.spread import fit_hedge_ratio, compute_spread

    alpha, beta = fit_hedge_ratio(synthetic_prices["Y"], synthetic_prices["X"])
    spread = compute_spread(synthetic_prices["Y"], synthetic_prices["X"], alpha, beta)
    p = engle_granger_adf_pvalue(spread)
    assert p < 0.10  # should reject non-stationarity


def test_adf_random_walk():
    """Random walk should have high ADF p-value (majority of seeds)."""
    # Use a drift term to make unit root more obvious
    rng = np.random.default_rng(123)
    rw = pd.Series(100.0 + np.cumsum(rng.normal(0.01, 1, 2000)))
    p = engle_granger_adf_pvalue(rw)
    assert p > 0.01


def test_estimate_half_life(synthetic_prices):
    from sarb.features.spread import fit_hedge_ratio, compute_spread

    alpha, beta = fit_hedge_ratio(synthetic_prices["Y"], synthetic_prices["X"])
    spread = compute_spread(synthetic_prices["Y"], synthetic_prices["X"], alpha, beta)
    hl = estimate_half_life(spread)
    assert np.isfinite(hl)
    assert hl > 0


def test_bootstrap_mean_ci():
    rng = np.random.default_rng(42)
    r = pd.Series(rng.normal(0.001, 0.01, 252))
    ci = bootstrap_mean_ci(r, n_boot=500)
    assert "mean" in ci
    assert "ci_low" in ci
    assert "ci_high" in ci
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]


def test_bootstrap_sharpe_ci():
    rng = np.random.default_rng(42)
    r = pd.Series(rng.normal(0.001, 0.01, 252))
    ci = bootstrap_sharpe_ci(r, n_boot=500)
    assert "sharpe" in ci
    assert "ci_low" in ci
    assert "ci_high" in ci
    assert ci["ci_low"] <= ci["sharpe"] <= ci["ci_high"]


def test_benjamini_hochberg_basic():
    # Known p-values: first two should be rejected at q=0.10
    pvals = [0.001, 0.01, 0.5, 0.8]
    mask = benjamini_hochberg(pvals, q=0.10)
    assert len(mask) == 4
    assert mask[0] is True
    assert mask[1] is True
    assert mask[3] is False


def test_benjamini_hochberg_empty():
    assert benjamini_hochberg([], q=0.10) == []


def test_benjamini_hochberg_all_high():
    pvals = [0.5, 0.6, 0.7]
    mask = benjamini_hochberg(pvals, q=0.05)
    assert all(m is False for m in mask)
