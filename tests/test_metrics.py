from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sarb.metrics.performance import (
    cagr,
    max_drawdown,
    max_drawdown_duration,
    performance_summary,
    sharpe,
    sortino,
    trade_statistics,
)


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


def test_sortino_ignores_upside_volatility():
    r = pd.Series([0.02, -0.01, 0.03, -0.01, 0.05])
    assert sortino(r) > sharpe(r)


def test_max_drawdown_duration():
    eq = pd.Series([1.0, 0.9, 0.95, 1.01, 0.99, 0.98, 1.02])
    assert max_drawdown_duration(eq) == 2


def test_trade_statistics():
    pos = pd.Series([0, 1, 1, 1, 0, -1, -1, 0], dtype=float)
    ret = pd.Series([0, 0.01, 0.01, -0.005, 0, -0.02, 0.005, 0])
    st = trade_statistics(pos, ret)
    assert st["n_trades"] == 2
    assert st["hit_rate"] == pytest.approx(0.5)
    assert st["avg_holding_days"] == pytest.approx(2.5)


def test_performance_summary_on_backtest(universe):
    from sarb.backtest.walkforward import walkforward_pairs_backtest

    bt = walkforward_pairs_backtest(universe, "B", "A", 250, 40, 1.5, 0.3, 1.0, 0.5)
    s = performance_summary(bt)
    for key in (
        "sharpe",
        "sortino",
        "cagr",
        "max_drawdown",
        "ann_turnover",
        "time_in_market",
        "n_trades",
        "hit_rate",
        "gross_sharpe",
        "cost_drag_ann",
    ):
        assert key in s
    assert s["gross_sharpe"] >= s["sharpe"]
    assert 0 < s["time_in_market"] < 1
