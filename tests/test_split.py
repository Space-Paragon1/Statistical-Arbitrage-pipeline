from __future__ import annotations
import pytest
import pandas as pd

from sarb.split.time_split import time_train_val_test_split
from sarb.split.rebalance import rolling_windows_by_quarter


def test_time_train_val_test_split(synthetic_prices):
    train, val, test = time_train_val_test_split(synthetic_prices, 0.6, 0.2)
    n = len(synthetic_prices)

    assert len(train) == int(n * 0.6)
    assert len(val) == int(n * 0.8) - int(n * 0.6)
    assert len(test) == n - int(n * 0.8)

    # No overlap
    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()


def test_time_train_val_test_split_invalid():
    df = pd.DataFrame({"a": range(100)})
    with pytest.raises(ValueError):
        time_train_val_test_split(df, 0.8, 0.3)


def test_rolling_windows_by_quarter(synthetic_prices):
    windows = rolling_windows_by_quarter(synthetic_prices, train_days=200, val_days=50)
    # With 500 days, we should get at least 1 window
    assert len(windows) >= 1

    for train_idx, val_idx, trade_idx in windows:
        assert len(train_idx) > 0
        assert len(val_idx) > 0
        assert len(trade_idx) >= 20
        # Time ordering: train < val < trade
        assert train_idx.max() <= val_idx.min()
        assert val_idx.max() <= trade_idx.min()
