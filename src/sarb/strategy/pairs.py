from __future__ import annotations
import pandas as pd

def generate_spread_positions(z: pd.Series, entry_z: float, exit_z: float) -> pd.Series:
    """
    Position in spread:
      +1 = long spread (long y, short x)
      -1 = short spread (short y, long x)
       0 = flat
    """
    pos = pd.Series(index=z.index, data=0.0)

    state = 0.0
    for t in range(len(z)):
        zi = z.iloc[t]
        if pd.isna(zi):
            pos.iloc[t] = state
            continue

        if state == 0.0:
            if zi <= -entry_z:
                state = +1.0
            elif zi >= entry_z:
                state = -1.0
        else:
            if abs(zi) <= exit_z:
                state = 0.0

        pos.iloc[t] = state

    # shift by 1 to trade next day (avoid lookahead)
    return pos.shift(1).fillna(0.0)
