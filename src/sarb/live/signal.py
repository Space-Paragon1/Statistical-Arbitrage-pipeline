from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

from sarb.features.spread import fit_hedge_ratio, compute_spread, rolling_zscore
from sarb.strategy.pairs import generate_spread_positions


@dataclass(frozen=True)
class PairSignal:
    y_ticker: str
    x_ticker: str
    z_score: float
    position: float  # +1, -1, or 0
    alpha: float
    beta: float
    timestamp: pd.Timestamp


def generate_live_signal(
    prices: pd.DataFrame,
    y: str,
    x: str,
    lookback_z: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    train_lookback: int = 504,
    hedge_method: str = "ols",
) -> PairSignal:
    """
    Generate signal from the latest available data.
    Uses the last train_lookback days to fit hedge ratio,
    then computes current z-score and position.
    """
    px = prices[[y, x]].dropna().tail(train_lookback + lookback_z)
    if len(px) < train_lookback:
        return PairSignal(
            y_ticker=y, x_ticker=x, z_score=0.0, position=0.0,
            alpha=0.0, beta=0.0, timestamp=px.index[-1],
        )

    train = px.iloc[:train_lookback]

    if hedge_method == "kalman":
        from sarb.features.kalman import fit_hedge_ratio_kalman
        alpha, beta = fit_hedge_ratio_kalman(train[y], train[x])
    else:
        alpha, beta = fit_hedge_ratio(train[y], train[x])

    spread = compute_spread(px[y], px[x], alpha, beta)
    z = rolling_zscore(spread, lookback_z)
    pos = generate_spread_positions(z, entry_z, exit_z)

    return PairSignal(
        y_ticker=y,
        x_ticker=x,
        z_score=float(z.iloc[-1]) if pd.notna(z.iloc[-1]) else 0.0,
        position=float(pos.iloc[-1]),
        alpha=alpha,
        beta=beta,
        timestamp=px.index[-1],
    )
