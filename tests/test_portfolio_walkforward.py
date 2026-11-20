from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sarb.research.select_pairs import scan_pairs_report
from sarb.research.walkforward_portfolio import (
    PortfolioBacktest,
    WFConfig,
    walkforward_quarterly_portfolio,
)
from sarb.risk.limits import RiskLimits
from sarb.split.rebalance import rolling_windows_by_quarter
from sarb.split.time_split import time_train_val_test_split


@pytest.fixture
def windows(universe):
    return rolling_windows_by_quarter(universe, train_days=300, val_days=100)


def _cfg(**kw):
    base = dict(top_k=2, corr_threshold=0.0, fdr_q=0.2, z_lookback=40, entry_z=1.5)
    base.update(kw)
    return WFConfig(**base)


def test_portfolio_runs_and_reports(universe, windows):
    res = walkforward_quarterly_portfolio(universe, list(universe.columns), windows, _cfg())
    assert isinstance(res, PortfolioBacktest)
    # Output spans exactly the walk-forward period (no zero-padded warm-up).
    assert res.returns.index[0] == windows[0][2][0]
    assert res.returns.index[-1] == windows[-1][2][-1]
    assert {"quarter_start", "n_candidates", "n_fdr_pass", "n_pairs", "breaker"} <= set(res.meta)
    assert res.meta["n_pairs"].sum() > 0
    # Cointegrated pairs A/B and C/D should dominate selection; E is noise.
    picked = ",".join(res.meta["pairs"])
    assert "A/B" in picked or "B/A" in picked
    assert np.isfinite(res.returns["equity"]).all()


def test_costs_charged_at_quarter_boundaries(universe, windows):
    """Asset-level simulation must charge for every weight change, including
    the flattening of positions when a pair is dropped at a quarter boundary."""
    res = walkforward_quarterly_portfolio(
        universe, list(universe.columns), windows, _cfg(fee_bps=5.0, slippage_bps=0.0)
    )
    w = res.weights
    traded = w.diff().abs().sum(axis=1) > 1e-6
    assert (res.returns.loc[traded, "costs"] > 0).all()


def test_gross_exposure_cap(universe, windows):
    cfg = _cfg(
        target_daily_vol=0.05,
        max_pair_scale=10.0,
        risk_limits=RiskLimits(
            max_position_size=1.0, max_gross_exposure=0.75, max_drawdown_pct=0.99
        ),
    )
    res = walkforward_quarterly_portfolio(universe, list(universe.columns), windows, cfg)
    assert res.weights.abs().sum(axis=1).max() <= 0.75 + 1e-9


def test_drawdown_breaker_rearms(universe, windows):
    """With an absurdly tight limit the breaker trips, but trading must resume
    in later quarters (the old implementation stayed off forever)."""
    cfg = _cfg(risk_limits=RiskLimits(max_position_size=1.0, max_drawdown_pct=0.0001))
    res = walkforward_quarterly_portfolio(universe, list(universe.columns), windows, cfg)
    flags = res.meta["breaker"].tolist()
    assert any(flags)
    first = flags.index(True)
    assert not all(flags[first:]), "breaker never re-armed"


def test_scan_report_strict_fdr(universe):
    train, val, _ = time_train_val_test_split(universe, 0.5, 0.2)
    rep = scan_pairs_report(
        universe,
        ["A", "E"],
        train.index,
        val.index,
        40,
        1.5,
        0.3,
        1.0,
        0.5,
        corr_threshold=0.0,
        fdr_q=0.01,
    )
    assert rep.n_fdr_pass == 0 and rep.selected == []
    rep_fb = scan_pairs_report(
        universe,
        ["A", "E"],
        train.index,
        val.index,
        40,
        1.5,
        0.3,
        1.0,
        0.5,
        corr_threshold=0.0,
        fdr_q=0.01,
        fallback_to_lowest_p=True,
    )
    assert len(rep_fb.selected) == 1


def test_scan_report_attaches_dsr(universe):
    train, val, _ = time_train_val_test_split(universe, 0.5, 0.2)
    rep = scan_pairs_report(
        universe,
        list(universe.columns),
        train.index,
        val.index,
        40,
        1.5,
        0.3,
        1.0,
        0.5,
        corr_threshold=0.0,
        fdr_q=0.2,
    )
    assert rep.selected
    assert all(0.0 <= r.val_dsr <= 1.0 for r in rep.selected)
    assert all(r.fdr_pass for r in rep.selected)
    assert isinstance(pd.DataFrame([vars(r) for r in rep.tested]), pd.DataFrame)
