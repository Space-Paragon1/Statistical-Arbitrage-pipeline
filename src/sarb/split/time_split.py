from __future__ import annotations
import pandas as pd

def time_train_val_test_split(
    df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not (0 < train_frac < 1) or not (0 < val_frac < 1) or (train_frac + val_frac) >= 1:
        raise ValueError("Need 0<train_frac<1, 0<val_frac<1, and train_frac+val_frac<1.")

    n = len(df)
    i1 = int(n * train_frac)
    i2 = int(n * (train_frac + val_frac))
    train = df.iloc[:i1].copy()
    val = df.iloc[i1:i2].copy()
    test = df.iloc[i2:].copy()
    return train, val, test
