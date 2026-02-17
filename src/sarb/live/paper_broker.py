from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime

from sarb.live.broker import BaseBroker, Order, Fill


@dataclass
class PaperBroker(BaseBroker):
    """Paper trading broker that simulates fills with slippage and fees."""

    initial_capital: float = 100_000.0
    fee_bps: float = 1.0
    slippage_bps: float = 0.5

    _cash: float = field(init=False, default=0.0)
    _positions: dict[str, float] = field(init=False, default_factory=dict)
    _fills: list[Fill] = field(init=False, default_factory=list)
    _price_cache: dict[str, float] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._cash = self.initial_capital

    def set_current_prices(self, prices: dict[str, float]) -> None:
        self._price_cache = prices.copy()

    def submit_order(self, order: Order) -> Fill | None:
        mid = self._price_cache.get(order.ticker)
        if mid is None or mid <= 0:
            return None

        # Apply slippage
        slip = mid * (self.slippage_bps / 1e4)
        if order.quantity > 0:
            fill_price = mid + slip
        else:
            fill_price = mid - slip

        notional = abs(order.quantity * fill_price)
        commission = notional * (self.fee_bps / 1e4)

        # Update cash
        cost = order.quantity * fill_price + commission
        self._cash -= cost

        # Update position
        current = self._positions.get(order.ticker, 0.0)
        self._positions[order.ticker] = current + order.quantity

        # Clean zero positions
        if abs(self._positions[order.ticker]) < 1e-10:
            del self._positions[order.ticker]

        ts = order.timestamp or datetime.now()
        fill = Fill(
            ticker=order.ticker,
            quantity=order.quantity,
            fill_price=fill_price,
            timestamp=ts,
            commission=commission,
        )
        self._fills.append(fill)
        return fill

    def get_position(self, ticker: str) -> float:
        return self._positions.get(ticker, 0.0)

    def get_all_positions(self) -> dict[str, float]:
        return dict(self._positions)

    def get_account_value(self) -> float:
        pos_value = sum(
            qty * self._price_cache.get(tkr, 0.0)
            for tkr, qty in self._positions.items()
        )
        return self._cash + pos_value

    def get_fill_history(self) -> list[Fill]:
        return list(self._fills)
