from __future__ import annotations
import pandas as pd
import numpy as np

from sarb.strategy.pairs import generate_spread_positions


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
