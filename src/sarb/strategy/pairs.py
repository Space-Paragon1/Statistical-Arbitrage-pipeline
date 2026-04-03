from __future__ import annotations
import pandas as pd

def generate_spread_positions(z: pd.Series, entry_z: float, exit_z: float) -> pd.Series:
    """
    Position in spread:
      +1 = long spread (long y, short x)
      -1 = short spread (short y, long x)
       0 = flat
    """
    values: list[float] = []
    state = 0.0
    for zi in z:
        if not pd.isna(zi):
            if state == 0.0:
                if zi <= -entry_z:
                    state = 1.0
                elif zi >= entry_z:
                    state = -1.0
            else:
                if abs(zi) <= exit_z:
                    state = 0.0
        values.append(state)

    pos = pd.Series(values, index=z.index, dtype=float)
    # shift by 1 to trade next day (avoid lookahead)
    return pos.shift(1).fillna(0.0)
