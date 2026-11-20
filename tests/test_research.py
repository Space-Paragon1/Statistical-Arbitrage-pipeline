from __future__ import annotations

from sarb.research.select_pairs import PairResult, evaluate_pair_on_val, scan_pairs
from sarb.split.time_split import time_train_val_test_split


def test_evaluate_pair_on_val(synthetic_prices):
    train, val, _ = time_train_val_test_split(synthetic_prices, 0.6, 0.2)
    result = evaluate_pair_on_val(
        prices=synthetic_prices,
        y="Y",
        x="X",
        train_idx=train.index,
        val_idx=val.index,
        lookback_z=60,
        entry_z=2.0,
        exit_z=0.5,
        fee_bps=1.0,
        slippage_bps=0.5,
    )
    assert result is not None
    assert isinstance(result, PairResult)
    assert result.y == "Y"
    assert result.x == "X"


def test_scan_pairs_small_universe(synthetic_prices):
    train, val, _ = time_train_val_test_split(synthetic_prices, 0.6, 0.2)
    selected = scan_pairs(
        prices=synthetic_prices,
        tickers=["Y", "X", "Z"],
        train_idx=train.index,
        val_idx=val.index,
        lookback_z=60,
        entry_z=2.0,
        exit_z=0.5,
        fee_bps=1.0,
        slippage_bps=0.5,
        corr_threshold=0.3,  # low threshold for synthetic data
        max_pairs=50,
        fdr_q=0.20,
        top_k=3,
    )
    assert isinstance(selected, list)
    # Should find at least 1 pair (Y/X are cointegrated)
    assert len(selected) >= 1
    assert all(isinstance(r, PairResult) for r in selected)


def test_parameter_sweep_deflates_best_cell(universe):
    import numpy as np

    from sarb.research.sweep import parameter_sweep

    eval_idx = universe.index[600:]
    res = parameter_sweep(
        universe, "B", "A", [1.0, 1.5, 2.0], [20, 40, 60], eval_idx, exit_z=0.3, train_lookback=250
    )
    assert res.n_trials == 9
    assert res.sharpe.shape == (3, 3)
    assert res.best_sharpe == np.nanmax(res.sharpe.to_numpy())
    assert res.best_dsr <= res.best_psr
