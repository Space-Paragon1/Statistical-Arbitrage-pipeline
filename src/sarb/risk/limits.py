from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskLimits:
    max_position_size: float = 0.25       # max weight per pair
    max_gross_exposure: float = 2.0       # total gross leverage
    max_drawdown_pct: float = 0.15        # kill switch threshold
    short_borrow_cost_bps: float = 0.0    # annualized borrow cost in bps


def apply_position_limits(
    weights: pd.Series,
    limits: RiskLimits,
) -> pd.Series:
    """Clip weights to max_position_size, rescale if gross > max_gross_exposure."""
    w = weights.clip(-limits.max_position_size, limits.max_position_size)
    gross = w.abs().sum()
    if gross > limits.max_gross_exposure:
        w = w * (limits.max_gross_exposure / gross)
    return w


def check_drawdown_breaker(
    equity: pd.Series,
    limits: RiskLimits,
) -> bool:
    """Returns True if max drawdown from peak exceeds limit (kill switch)."""
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return bool(dd.min() < -limits.max_drawdown_pct)


def compute_short_borrow_cost(
    short_notional: pd.Series,
    borrow_cost_bps: float,
    ann_factor: int = 252,
) -> pd.Series:
    """Daily cost of carrying short positions."""
    daily_rate = borrow_cost_bps / 1e4 / ann_factor
    return short_notional.abs() * daily_rate
