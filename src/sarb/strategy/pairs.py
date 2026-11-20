"""Z-score mean-reversion rules for a spread.

Position convention: ``+1`` long spread (long y, short beta*x), ``-1`` short spread,
``0`` flat. The same :func:`next_state` transition is used by the vectorised
backtest and by the live runner, so the two cannot drift apart.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalRules:
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float | None = None  # exit if |z| blows out beyond this (relationship break)
    max_holding: int | None = None  # time stop, in bars

    def __post_init__(self) -> None:
        if not 0 <= self.exit_z < self.entry_z:
            raise ValueError("Need 0 <= exit_z < entry_z.")
        if self.stop_z is not None and self.stop_z <= self.entry_z:
            raise ValueError("stop_z must exceed entry_z.")
        if self.max_holding is not None and self.max_holding < 1:
            raise ValueError("max_holding must be >= 1.")


@dataclass
class PositionState:
    position: float = 0.0
    bars_held: int = 0
    # After a stop-out, block re-entry until z comes back inside the entry band.
    locked_out: bool = False


def next_state(state: PositionState, z: float, rules: SignalRules) -> PositionState:
    """One-step transition given the z-score observed at the close."""
    if z is None or math.isnan(z):
        # No information: hold whatever we have.
        held = state.bars_held + 1 if state.position != 0.0 else 0
        return PositionState(state.position, held, state.locked_out)

    az = abs(z)
    if state.position == 0.0:
        locked = state.locked_out and az >= rules.entry_z
        if not locked:
            if z <= -rules.entry_z:
                return PositionState(1.0, 1, False)
            if z >= rules.entry_z:
                return PositionState(-1.0, 1, False)
        return PositionState(0.0, 0, locked)

    if rules.stop_z is not None and az >= rules.stop_z:
        return PositionState(0.0, 0, True)
    # One-sided exit: take profit once the spread has reverted to within exit_z
    # of the mean *or overshot through it* (a plain |z| <= exit_z test would keep
    # a long open if z jumped straight from -2 to +2.5).
    reverted = z >= -rules.exit_z if state.position > 0 else z <= rules.exit_z
    if reverted:
        return PositionState(0.0, 0, False)
    if rules.max_holding is not None and state.bars_held >= rules.max_holding:
        return PositionState(0.0, 0, True)
    return PositionState(state.position, state.bars_held + 1, False)


def spread_states(z: pd.Series, rules: SignalRules) -> pd.Series:
    """Position decided at each close (NOT shifted). Index-aligned with ``z``."""
    out = np.empty(len(z), dtype=float)
    st = PositionState()
    for i, zi in enumerate(z.to_numpy(dtype=float)):
        st = next_state(st, zi, rules)
        out[i] = st.position
    return pd.Series(out, index=z.index, dtype=float)


def generate_spread_positions(
    z: pd.Series,
    entry_z: float,
    exit_z: float,
    stop_z: float | None = None,
    max_holding: int | None = None,
) -> pd.Series:
    """Position *held* on each bar: the state decided at the previous close.

    The one-bar shift is what makes the backtest lookahead-safe: the z-score at
    ``t`` (which uses the close at ``t``) can only affect the return earned from
    ``t`` to ``t+1``.
    """
    rules = SignalRules(entry_z=entry_z, exit_z=exit_z, stop_z=stop_z, max_holding=max_holding)
    return spread_states(z, rules).shift(1).fillna(0.0)
