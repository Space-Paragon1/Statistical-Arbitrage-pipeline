from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sarb.strategy.pairs import SignalRules, generate_spread_positions, spread_states


def test_generate_spread_positions_basic():
    # Hand-crafted z-score series
    z = pd.Series([0.0, -2.5, -2.0, -0.3, 0.0, 2.5, 2.0, 0.3, 0.0])
    pos = generate_spread_positions(z, entry_z=2.0, exit_z=0.5)

    assert isinstance(pos, pd.Series)
    assert len(pos) == len(z)
    # All positions should be in {-1, 0, +1}
    assert set(pos.unique()).issubset({-1.0, 0.0, 1.0})


def test_generate_spread_positions_shift():
    """Positions should be shifted by 1 day (no lookahead)."""
    z = pd.Series([0.0, -3.0, -3.0, 0.0, 0.0])
    pos = generate_spread_positions(z, entry_z=2.0, exit_z=0.5)

    # First position should always be 0 due to shift
    assert pos.iloc[0] == 0.0


def test_generate_spread_positions_entry_exit():
    """Test that positions enter on extreme z and exit near zero."""
    z = pd.Series([0.0, 0.0, -3.0, -2.5, -0.2, 0.0, 3.0, 2.5, 0.1, 0.0])
    pos = generate_spread_positions(z, entry_z=2.0, exit_z=0.5)

    # After seeing z=-3.0 at index 2, position should be +1 at index 3 (shifted)
    assert pos.iloc[3] == 1.0
    # After z crosses back to |z|<=0.5, should exit
    # The position at index 5 (from z=-0.2 at index 4) should be 0
    assert pos.iloc[5] == 0.0


def test_overshoot_through_band_exits():
    """A long entered at z=-3 must close if z jumps straight to +2.5 (a plain
    |z| <= exit_z rule would keep it open). The next bar is a fresh short entry."""
    z = pd.Series([0.0, -3.0, 2.5, 2.4, 0.3])
    states = spread_states(z, SignalRules(entry_z=2.0, exit_z=0.5))
    assert states.tolist() == [0.0, 1.0, 0.0, -1.0, 0.0]


def test_stop_loss_and_lockout():
    z = pd.Series([0.0, -2.5, -3.5, -4.5, -3.0, -2.5, -1.0, -2.2])
    states = spread_states(z, SignalRules(entry_z=2.0, exit_z=0.5, stop_z=4.0))
    # stopped out at -4.5; no re-entry at -3.0/-2.5 (still outside the entry band);
    # re-armed once z is back inside the band (-1.0), then re-enters at -2.2.
    assert states.tolist() == [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def test_time_stop():
    z = pd.Series([-2.5, -2.4, -2.3, -2.2, -2.1, -1.0, -2.5])
    states = spread_states(z, SignalRules(entry_z=2.0, exit_z=0.5, max_holding=3))
    assert states.tolist() == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def test_nan_z_holds_position():
    z = pd.Series([-2.5, np.nan, -2.0, 0.0])
    assert spread_states(z, SignalRules()).tolist() == [1.0, 1.0, 1.0, 0.0]


@pytest.mark.parametrize(
    "kw", [dict(entry_z=1.0, exit_z=1.5), dict(stop_z=1.5), dict(max_holding=0)]
)
def test_invalid_rules_rejected(kw):
    with pytest.raises(ValueError):
        SignalRules(**kw)
