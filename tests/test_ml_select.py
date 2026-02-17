from __future__ import annotations
import numpy as np
import pandas as pd

from sarb.research.ml_select import (
    cluster_pairs_optics,
    cluster_pairs_dbscan,
    ml_prefilter_pairs,
    ClusterPairScore,
)


def _make_4ticker_prices():
    """4 tickers: A/B correlated, C/D correlated, A/C uncorrelated."""
    rng = np.random.default_rng(42)
    n = 300
    dates = pd.bdate_range("2020-01-01", periods=n)
    base1 = np.cumsum(rng.normal(0, 1, n))
    base2 = np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "A": 100 + base1 + rng.normal(0, 0.3, n),
            "B": 100 + base1 + rng.normal(0, 0.3, n),
            "C": 50 + base2 + rng.normal(0, 0.3, n),
            "D": 50 + base2 + rng.normal(0, 0.3, n),
        },
        index=dates,
    )


def test_cluster_pairs_optics():
    px = _make_4ticker_prices()
    ret = px.pct_change().dropna()
    pairs = cluster_pairs_optics(ret, min_samples=2)
    assert isinstance(pairs, list)
    assert all(isinstance(p, ClusterPairScore) for p in pairs)
    # Should find at least some pairs
    assert len(pairs) >= 1


def test_cluster_pairs_dbscan():
    px = _make_4ticker_prices()
    ret = px.pct_change().dropna()
    pairs = cluster_pairs_dbscan(ret, eps=2.0, min_samples=2)
    assert isinstance(pairs, list)


def test_ml_prefilter_pairs():
    px = _make_4ticker_prices()
    train_idx = px.index[:200]
    result = ml_prefilter_pairs(
        prices=px, tickers=list(px.columns), train_idx=train_idx,
        method="optics", max_pairs=10,
    )
    assert isinstance(result, list)
    assert len(result) >= 1
    assert all(isinstance(p, tuple) and len(p) == 2 for p in result)
