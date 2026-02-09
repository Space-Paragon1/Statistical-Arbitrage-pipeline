from __future__ import annotations
import pandas as pd

def time_train_test_split(df: pd.DataFrame, train_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not (0.0 < train_frac < 1.0):
        raise ValueError("train_frac must be between 0 and 1.")
    n = len(df)
    cut = int(n * train_frac)
    train = df.iloc[:cut].copy()
    test = df.iloc[cut:].copy()
    return train, test
