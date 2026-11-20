"""Portfolio risk limits."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RiskLimits:
    max_position_size: float = 0.25  # max allocation per pair (fraction of capital)
    max_gross_exposure: float = 2.0  # cap on sum |asset weights|, enforced daily
    max_drawdown_pct: float = 0.15  # kill switch threshold on *current* drawdown


def apply_position_limits(weights: pd.Series, limits: RiskLimits) -> pd.Series:
    """Clip weights to ``max_position_size`` and rescale if gross exceeds the cap."""
    w = weights.clip(-limits.max_position_size, limits.max_position_size)
    gross = w.abs().sum()
    if gross > limits.max_gross_exposure:
        w = w * (limits.max_gross_exposure / gross)
    return w


def cap_gross_exposure(weights: pd.DataFrame, max_gross: float) -> pd.DataFrame:
    """Scale each row of an asset-weight matrix down so ``sum |w| <= max_gross``."""
    gross = weights.abs().sum(axis=1)
    scale = (max_gross / gross).where(gross > max_gross, 1.0)
    return weights.mul(scale, axis=0)


def current_drawdown(equity: pd.Series) -> float:
    """Drawdown of the latest equity value from its running peak (<= 0)."""
    eq = equity.dropna()
    if eq.empty:
        return 0.0
    return float(eq.iloc[-1] / eq.max() - 1.0)


def check_drawdown_breaker(equity: pd.Series, limits: RiskLimits) -> bool:
    """True if the *current* drawdown breaches the limit.

    Uses the live drawdown rather than the worst historical one: a breaker keyed
    to the historical maximum could never re-arm, because after flattening the
    equity curve stops moving and the old trough is remembered forever.
    """
    return current_drawdown(equity) < -limits.max_drawdown_pct


def compute_short_borrow_cost(
    short_notional: pd.Series,
    borrow_cost_bps: float,
    ann_factor: int = 252,
) -> pd.Series:
    """Daily cost of carrying short positions."""
    daily_rate = borrow_cost_bps / 1e4 / ann_factor
    return short_notional.abs() * daily_rate
