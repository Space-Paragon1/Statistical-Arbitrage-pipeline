from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sarb.stats.bootstrap import (
    bootstrap_mean_ci,
    bootstrap_sharpe_ci,
    stationary_bootstrap_indices,
)
from sarb.stats.cointegration import (
    engle_granger_adf_pvalue,
    engle_granger_test,
    estimate_half_life,
)
from sarb.stats.multiple_testing import benjamini_hochberg
from sarb.stats.sharpe_inference import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)


def test_adf_stationary(synthetic_prices):
    """Spread of cointegrated pair should have low ADF p-value."""
    from sarb.features.spread import compute_spread, fit_hedge_ratio

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
    from sarb.features.spread import compute_spread, fit_hedge_ratio

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


def test_engle_granger_detects_cointegration(synthetic_prices):
    res = engle_granger_test(synthetic_prices["Y"], synthetic_prices["X"])
    assert res.pvalue < 0.01
    assert res.beta == pytest.approx(1.2, abs=0.15)


def test_engle_granger_rejects_independent_random_walks():
    rejections = 0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        y = pd.Series(np.cumsum(rng.normal(size=600)))
        x = pd.Series(np.cumsum(rng.normal(size=600)))
        rejections += engle_granger_test(y, x).pvalue < 0.05
    assert rejections <= 3  # ~5% size; allow sampling noise


def test_engle_granger_is_more_conservative_than_naive_adf():
    """Plain ADF on fitted residuals over-rejects; MacKinnon EG p-values must be larger."""
    from sarb.features.spread import compute_spread

    rng = np.random.default_rng(11)
    y = pd.Series(np.cumsum(rng.normal(size=500)))
    x = pd.Series(np.cumsum(rng.normal(size=500)))
    eg = engle_granger_test(y, x)
    naive = engle_granger_adf_pvalue(compute_spread(y, x, eg.alpha, eg.beta))
    assert eg.pvalue > naive


def test_half_life_recovers_ou_parameter():
    rng = np.random.default_rng(5)
    phi, n = 0.95, 20_000
    s = np.zeros(n)
    for i in range(1, n):
        s[i] = phi * s[i - 1] + rng.normal()
    expected = -np.log(2) / np.log(phi)  # ~13.5 days
    assert estimate_half_life(pd.Series(s)) == pytest.approx(expected, rel=0.1)


def test_benjamini_hochberg_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests

    rng = np.random.default_rng(0)
    p = np.concatenate([rng.uniform(0, 0.01, 10), rng.uniform(0, 1, 90)])
    ours = benjamini_hochberg(list(p), q=0.1)
    ref = multipletests(p, alpha=0.1, method="fdr_bh")[0]
    assert ours == ref.tolist()


def test_stationary_bootstrap_indices():
    rng = np.random.default_rng(0)
    idx = stationary_bootstrap_indices(500, 200, mean_block=10.0, rng=rng)
    assert idx.shape == (200, 500)
    assert idx.min() >= 0 and idx.max() < 500
    # Consecutive indices (mod n) continue a block with probability 1 - 1/10.
    cont = ((idx[:, 1:] - idx[:, :-1]) % 500 == 1).mean()
    assert cont == pytest.approx(0.9, abs=0.02)


def test_block_bootstrap_is_wider_for_autocorrelated_returns():
    rng = np.random.default_rng(2)
    n = 1000
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = 0.6 * e[i - 1] + rng.normal(0.0003, 0.01)
    r = pd.Series(e)
    iid = bootstrap_sharpe_ci(r, n_boot=1000, method="iid")
    blk = bootstrap_sharpe_ci(r, n_boot=1000, method="stationary", block_size=20)
    assert (blk["ci_high"] - blk["ci_low"]) > (iid["ci_high"] - iid["ci_low"])


def test_probabilistic_and_deflated_sharpe():
    rng = np.random.default_rng(9)
    strong = pd.Series(rng.normal(0.002, 0.01, 1000))  # ~3.2 annual Sharpe
    assert probabilistic_sharpe_ratio(strong) > 0.99
    noise = pd.Series(rng.normal(0.0, 0.01, 1000))
    assert 0.05 < probabilistic_sharpe_ratio(noise) < 0.95

    var = 0.05**2
    assert expected_max_sharpe(100, var) > expected_max_sharpe(10, var) > 0
    dsr = deflated_sharpe_ratio(strong, n_trials=100, sr_variance=var)
    assert dsr < probabilistic_sharpe_ratio(strong)
