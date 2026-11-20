"""Lookahead-bias invariance tests.

The strongest check a backtest can have: corrupt every price strictly after a
cut-off date and require that *everything* the strategy reported up to that date
is bit-for-bit unchanged. Any leak of future information fails these tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sarb.backtest.walkforward import walkforward_pairs_backtest
from sarb.live.signal import generate_live_signal
from sarb.research.walkforward_portfolio import WFConfig, walkforward_quarterly_portfolio
from sarb.risk.limits import RiskLimits
from sarb.split.rebalance import rolling_windows_by_quarter


def _perturb_after(prices: pd.DataFrame, cutoff: pd.Timestamp, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = prices.copy()
    future = out.index > cutoff
    shock = np.exp(np.cumsum(rng.normal(0, 0.05, (future.sum(), out.shape[1])), axis=0))
    out.loc[future] = out.loc[future].to_numpy() * shock
    return out


@pytest.mark.parametrize("hedge_method", ["ols", "kalman"])
def test_walkforward_pair_has_no_lookahead(universe, hedge_method):
    cutoff = universe.index[700]
    kwargs = dict(
        y="B",
        x="A",
        train_lookback=250,
        z_lookback=40,
        entry_z=1.5,
        exit_z=0.3,
        fee_bps=1.0,
        slippage_bps=0.5,
        hedge_method=hedge_method,
        stop_z=4.0,
    )
    base = walkforward_pairs_backtest(universe, **kwargs)
    pert = walkforward_pairs_backtest(_perturb_after(universe, cutoff), **kwargs)

    assert base["pos"].abs().sum() > 0, "strategy never traded; test would be vacuous"
    cols = ["alpha", "beta", "z", "pos", "w_y", "w_x", "ret_net", "costs", "equity"]
    pd.testing.assert_frame_equal(base.loc[:cutoff, cols], pert.loc[:cutoff, cols])
    # Sanity: the perturbation does change the future.
    assert not np.allclose(
        base.loc[cutoff:, "ret_net"].iloc[2:], pert.loc[cutoff:, "ret_net"].iloc[2:]
    )


def test_portfolio_walkforward_has_no_lookahead(universe):
    windows = rolling_windows_by_quarter(universe, train_days=300, val_days=100)
    cfg = WFConfig(
        top_k=2,
        corr_threshold=0.0,
        fdr_q=0.2,
        z_lookback=40,
        entry_z=1.5,
        use_correlation_weights=True,
        risk_limits=RiskLimits(max_position_size=0.6, max_drawdown_pct=0.5),
    )
    cutoff = windows[len(windows) // 2][2][10]  # mid-quarter

    base = walkforward_quarterly_portfolio(universe, list(universe.columns), windows, cfg)
    pert = walkforward_quarterly_portfolio(
        _perturb_after(universe, cutoff), list(universe.columns), windows, cfg
    )
    assert (base.meta["n_pairs"] > 0).any(), "portfolio never traded; test would be vacuous"
    pd.testing.assert_frame_equal(base.returns.loc[:cutoff], pert.returns.loc[:cutoff])
    pd.testing.assert_frame_equal(base.weights.loc[:cutoff], pert.weights.loc[:cutoff])


def test_live_signal_matches_backtest(universe):
    """Backtest/live parity: replaying history through the live signal path must
    reproduce the backtest's decisions exactly."""
    kwargs = dict(train_lookback=250, entry_z=1.5, exit_z=0.3)
    bt = walkforward_pairs_backtest(
        universe, "B", "A", z_lookback=40, fee_bps=0, slippage_bps=0, **kwargs
    )
    decided = bt["pos"].shift(-1)  # position decided at close t is held over t+1
    for i in range(300, len(universe) - 1, 37):
        sig = generate_live_signal(universe.iloc[: i + 1], "B", "A", lookback_z=40, **kwargs)
        assert sig.position == decided.iloc[i], universe.index[i]
        assert sig.z_score == pytest.approx(bt["z"].iloc[i])
        # The hedge ratio the live process holds overnight is the backtest's next-bar beta.
        assert sig.beta == pytest.approx(bt["beta"].iloc[i + 1], rel=1e-9)
