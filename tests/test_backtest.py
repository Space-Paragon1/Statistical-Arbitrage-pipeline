from __future__ import annotations
import pandas as pd

from sarb.features.spread import fit_hedge_ratio, compute_spread, rolling_zscore
from sarb.strategy.pairs import generate_spread_positions
from sarb.backtest.engine import backtest_pairs


def test_backtest_pairs(synthetic_prices):
    y, x = "Y", "X"
    alpha, beta = fit_hedge_ratio(synthetic_prices[y], synthetic_prices[x])
    spread = compute_spread(synthetic_prices[y], synthetic_prices[x], alpha, beta)
    z = rolling_zscore(spread, 60)
    pos = generate_spread_positions(z, entry_z=2.0, exit_z=0.5)

    bt = backtest_pairs(
        prices=synthetic_prices[[y, x]],
        y=y, x=x,
        alpha=alpha, beta=beta,
        spread_pos=pos,
        fee_bps=1.0, slippage_bps=0.5,
        leverage=1.0,
    )

    assert isinstance(bt, pd.DataFrame)
    assert "equity" in bt.columns
    assert "ret_net" in bt.columns
    assert "costs" in bt.columns

    # Equity should start near 1.0
    first_valid = bt["equity"].dropna().iloc[0]
    assert abs(first_valid - 1.0) < 0.05

    # Costs should be non-negative
    assert (bt["costs"] >= -1e-10).all()


def test_backtest_pairs_flat_position(synthetic_prices):
    """With zero positions, equity should stay at 1.0 and costs at 0."""
    y, x = "Y", "X"
    pos = pd.Series(0.0, index=synthetic_prices.index)

    bt = backtest_pairs(
        prices=synthetic_prices[[y, x]],
        y=y, x=x,
        alpha=0.0, beta=1.0,
        spread_pos=pos,
        fee_bps=1.0, slippage_bps=0.5,
        leverage=1.0,
    )

    assert (bt["ret_net"] == 0.0).all()
