from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Order:
    ticker: str
    quantity: float  # positive = buy, negative = sell
    order_type: str = "market"
    limit_price: float | None = None
    timestamp: datetime | None = None


@dataclass(frozen=True)
class Fill:
    ticker: str
    quantity: float
    fill_price: float
    timestamp: datetime
    commission: float = 0.0


class BaseBroker(ABC):
    @abstractmethod
    def submit_order(self, order: Order) -> Fill | None:
        ...

    @abstractmethod
    def get_position(self, ticker: str) -> float:
        ...

    @abstractmethod
    def get_all_positions(self) -> dict[str, float]:
        ...

    @abstractmethod
    def get_account_value(self) -> float:
        ...
