from __future__ import annotations
import pandas as pd

def quarter_boundaries(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """
    Returns quarter start timestamps present in index.
    """
    idx = pd.DatetimeIndex(index).sort_values()
    # PeriodIndex gives quarter labels; we take first date seen in each quarter
    q = idx.to_period("Q")
    # groupby on Index can return a mapping-like object in newer pandas
    first_dates = idx.to_series().groupby(q).min()
    return list(first_dates.values)

def rolling_windows_by_quarter(
    prices: pd.DataFrame,
    train_days: int = 504,   # ~2 years
    val_days: int = 126,     # ~6 months
) -> list[tuple[pd.Index, pd.Index, pd.Index]]:
    """
    For each quarter start t:
      - train = [t - (train_days+val_days), t - val_days)
      - val   = [t - val_days, t)
      - trade = [t, next_quarter_start)
    Returns (train_idx, val_idx, trade_idx) windows. All time-safe.
    """
    idx = pd.DatetimeIndex(prices.index).sort_values()
    q_starts = quarter_boundaries(idx)

    windows = []
    for i, t0 in enumerate(q_starts[:-1]):
        t1 = q_starts[i + 1]
        # trade window
        trade_idx = idx[(idx >= t0) & (idx < t1)]
        if len(trade_idx) < 20:
            continue

        end_val = t0
        start_val = idx[idx < end_val][-val_days] if (idx < end_val).sum() >= val_days else None
        if start_val is None:
            continue
        val_idx = idx[(idx >= start_val) & (idx < end_val)]

        end_train = start_val
        if (idx < end_train).sum() < train_days:
            continue
        start_train = idx[idx < end_train][-train_days]
        train_idx = idx[(idx >= start_train) & (idx < end_train)]

        windows.append((train_idx, val_idx, trade_idx))

    return windows
