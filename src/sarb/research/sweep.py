"""Parameter sweeps with an honest multiple-testing haircut.

Sweeping a grid of signal parameters and reporting the best cell is the textbook
way to overfit a backtest. :func:`parameter_sweep` makes that visible: it reports
the whole grid *and* the Deflated Sharpe Ratio of the best cell, i.e. the
probability that its Sharpe beats what the best of ``N`` skill-less
configurations would show by luck.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sarb.backtest.walkforward import walkforward_pairs_backtest
from sarb.metrics.performance import sharpe
from sarb.stats.sharpe_inference import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    per_period_sharpe,
    probabilistic_sharpe_ratio,
)


@dataclass(frozen=True)
class SweepResult:
    sharpe: pd.DataFrame  # annualised Sharpe, index = z_lookback, columns = entry_z
    best: tuple[int, float]  # (z_lookback, entry_z) of the best cell
    best_sharpe: float
    best_psr: float  # P(true SR > 0) ignoring the search
    best_dsr: float  # same, deflated for the number of cells tried
    n_trials: int
    expected_max_sharpe_ann: float  # best annualised Sharpe expected from pure luck


def parameter_sweep(
    prices: pd.DataFrame,
    y: str,
    x: str,
    entry_grid: Sequence[float],
    lookback_grid: Sequence[int],
    eval_index: pd.Index,
    exit_z: float = 0.5,
    train_lookback: int = 504,
    fee_bps: float = 1.0,
    slippage_bps: float = 0.5,
    hedge_method: str = "ols",
) -> SweepResult:
    """Walk-forward backtest for every ``(z_lookback, entry_z)`` cell, scored on ``eval_index``."""
    rets: dict[tuple[int, float], pd.Series] = {}
    sharpes: dict[tuple[int, float], float] = {}
    for lb in lookback_grid:
        for ez in entry_grid:
            if exit_z >= ez:
                continue
            bt = walkforward_pairs_backtest(
                prices,
                y,
                x,
                train_lookback=train_lookback,
                z_lookback=lb,
                entry_z=ez,
                exit_z=exit_z,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                hedge_method=hedge_method,
            )
            r = bt["ret_net"].reindex(eval_index).fillna(0.0)
            rets[(lb, ez)] = r
            sharpes[(lb, ez)] = sharpe(r)

    table = (
        pd.Series(sharpes)
        .unstack()
        .reindex(index=list(lookback_grid), columns=list(entry_grid))
        .rename_axis(index="z_lookback", columns="entry_z")
    )
    best = max(sharpes, key=lambda k: sharpes[k])
    sr_trials = np.array([per_period_sharpe(r) for r in rets.values()])
    sr_trials = sr_trials[np.isfinite(sr_trials)]
    sr_var = float(np.var(sr_trials, ddof=1)) if len(sr_trials) > 1 else 0.0
    n = len(rets)
    return SweepResult(
        sharpe=table,
        best=best,
        best_sharpe=sharpes[best],
        best_psr=probabilistic_sharpe_ratio(rets[best]),
        best_dsr=deflated_sharpe_ratio(rets[best], n_trials=n, sr_variance=sr_var),
        n_trials=n,
        expected_max_sharpe_ann=expected_max_sharpe(n, sr_var) * np.sqrt(252),
    )
