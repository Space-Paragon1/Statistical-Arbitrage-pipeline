"""Exact, hand-checkable tests of the weight-based engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sarb.backtest.engine import backtest_pairs, backtest_weights, pair_weights


def _frame(rows, cols=("A", "B")):
    return pd.DataFrame(
        rows, columns=list(cols), index=pd.bdate_range("2024-01-01", periods=len(rows))
    )


def test_backtest_weights_hand_computed():
    w = _frame([[0.0, 0.0], [0.5, -0.5], [0.5, -0.5], [0.0, 0.0]])
    r = _frame([[0.0, 0.0], [0.02, 0.01], [-0.01, 0.03], [0.05, 0.05]])
    bt = backtest_weights(w, r, cost_bps=10.0)

    # Day 1: enter 0.5/-0.5 from flat -> turnover 1.0, gross = 0.5*0.02 - 0.5*0.01
    assert bt["ret_gross"].iloc[1] == pytest.approx(0.005)
    assert bt["turnover"].iloc[1] == pytest.approx(1.0)
    assert bt["costs"].iloc[1] == pytest.approx(1.0 * 10 / 1e4)

    # Day 2: drift-adjusted turnover. After day 1 the weights drifted to
    # [0.5*1.02, -0.5*1.01] / 1.005 before rebalancing back to [0.5, -0.5].
    drifted = np.array([0.5 * 1.02, -0.5 * 1.01]) / 1.005
    assert bt["turnover"].iloc[2] == pytest.approx(np.abs(np.array([0.5, -0.5]) - drifted).sum())

    # Day 3: flat, so no P&L; cost of closing the drifted book.
    assert bt["ret_gross"].iloc[3] == 0.0
    assert bt["gross_exposure"].tolist() == [0.0, 1.0, 1.0, 0.0]
    assert bt["equity"].iloc[-1] == pytest.approx(np.prod(1 + bt["ret_net"]))


def test_borrow_cost_is_charged_on_short_notional():
    w = _frame([[0.5, -0.5]] * 3)
    r = _frame([[0.0, 0.0]] * 3)
    bt = backtest_weights(w, r, cost_bps=0.0, short_borrow_cost_bps=252.0)
    # 252 bps/yr on 0.5 short = 0.5 * 1bp per day
    assert bt["costs"].iloc[1] == pytest.approx(0.5 * 1e-4)


def test_missing_return_with_open_position_raises():
    w = _frame([[0.5, -0.5], [0.5, -0.5]])
    r = _frame([[0.0, 0.0], [np.nan, 0.0]])
    with pytest.raises(ValueError, match="Missing returns"):
        backtest_weights(w, r)


def test_pair_pnl_equals_spread_pnl():
    """With price-level beta, the engine's P&L must equal the P&L of the spread
    y - beta*x, normalised by gross notional at the previous close."""
    rng = np.random.default_rng(0)
    n, beta, lev = 60, 0.35, 1.5
    idx = pd.bdate_range("2024-01-01", periods=n)
    py = pd.Series(40 + np.cumsum(rng.normal(0, 0.4, n)), index=idx, name="KO")
    px = pd.Series(110 + np.cumsum(rng.normal(0, 1.0, n)), index=idx, name="PEP")
    pos = pd.Series(np.where(np.arange(n) % 20 < 10, 1.0, -1.0), index=idx)
    pos.iloc[0] = 0.0

    prices = pd.concat([py, px], axis=1)
    bt = backtest_pairs(prices, "KO", "PEP", 0.0, beta, pos, 0.0, 0.0, leverage=lev)

    d_spread = py.diff() - beta * px.diff()
    gross_prev = py.shift(1) + beta * px.shift(1)
    expected = (lev * pos * d_spread / gross_prev).fillna(0.0)
    np.testing.assert_allclose(bt["ret_gross"].to_numpy(), expected.to_numpy(), atol=1e-12)


def test_pair_weights_dollar_ratio_uses_prices():
    idx = pd.bdate_range("2024-01-01", periods=3)
    py = pd.Series([40.0, 40.0, 40.0], index=idx, name="Y")
    px = pd.Series([100.0, 100.0, 100.0], index=idx, name="X")
    w = pair_weights(py, px, pd.Series([0.0, 1.0, 1.0], index=idx), beta=0.3, leverage=1.0)
    # dollars: +40 in Y, -30 in X -> weights 40/70, -30/70
    assert w["Y"].iloc[1] == pytest.approx(40 / 70)
    assert w["X"].iloc[1] == pytest.approx(-30 / 70)
    np.testing.assert_allclose(w.abs().sum(axis=1).iloc[1:], 1.0)
