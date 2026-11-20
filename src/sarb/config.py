"""Default research configuration."""

from __future__ import annotations

from dataclasses import dataclass

# Liquid US large caps and ETFs with full history over the default sample. Note the
# survivorship bias inherent in choosing today's well-known names for a 2015 start.
DEFAULT_UNIVERSE: tuple[str, ...] = (
    "KO",
    "PEP",
    "XOM",
    "CVX",
    "JPM",
    "BAC",
    "MSFT",
    "AAPL",
    "SPY",
    "IWM",
    "QQQ",
    "DIA",
    "GLD",
    "SLV",
    "TLT",
    "IEF",
)


@dataclass(frozen=True)
class PairConfig:
    y_ticker: str = "KO"
    x_ticker: str = "PEP"
    start: str = "2015-01-01"
    end: str = "2025-01-01"
    price_field: str = "Adj Close"

    train_frac: float = 0.7  # reporting split: metrics are quoted on the last 30%
    train_lookback: int = 504  # rolling hedge-ratio window (~2 years)
    hedge_method: str = "ols"

    lookback_z: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float | None = None
    max_holding: int | None = None

    # Trading assumptions
    fee_bps: float = 1.0  # per unit of traded notional
    slippage_bps: float = 0.5  # per unit of traded notional
    short_borrow_cost_bps: float = 0.0  # annualised, on short notional
    leverage: float = 1.0  # gross leverage
