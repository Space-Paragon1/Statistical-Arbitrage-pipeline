from dataclasses import dataclass

@dataclass(frozen=True)
class PairConfig:
    y_ticker: str = "KO"
    x_ticker: str = "PEP"
    start: str = "2015-01-01"
    end: str = "2025-01-01"
    price_field: str = "Adj Close"

    train_frac: float = 0.7  # time-aware split

    lookback_z: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5

    # Trading assumptions
    fee_bps: float = 1.0        # per trade leg (bps of notional)
    slippage_bps: float = 0.5   # per trade leg
    leverage: float = 1.0       # gross leverage target
