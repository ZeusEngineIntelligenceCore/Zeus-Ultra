#!/usr/bin/env python3
"""
ZEUS EXCHANGE BASE - Abstract Exchange Interface
Multi-exchange architecture for future expansion
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REJECTED = "rejected"


@dataclass
class Ticker:
    symbol: str
    bid: float
    ask: float
    last: float
    volume_24h: float
    high_24h: float
    low_24h: float
    timestamp: datetime


@dataclass
class OHLCV:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class OrderBook:
    symbol: str
    bids: List[tuple]
    asks: List[tuple]
    timestamp: datetime

    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0.0

    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else 0.0

    def spread(self) -> float:
        return self.best_ask() - self.best_bid()

    def spread_pct(self) -> float:
        mid = (self.best_ask() + self.best_bid()) / 2
        if mid == 0:
            return 0.0
        return (self.spread() / mid) * 100

    def bid_depth(self, levels: int = 5) -> float:
        return sum(price * size for price, size in self.bids[:levels])

    def ask_depth(self, levels: int = 5) -> float:
        return sum(price * size for price, size in self.asks[:levels])

    def imbalance(self, levels: int = 5) -> float:
        bid_vol = sum(size for _, size in self.bids[:levels])
        ask_vol = sum(size for _, size in self.asks[:levels])
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total


@dataclass
class Order:
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    price: Optional[float]
    amount: float
    filled: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    timestamp: datetime = field(default_factory=datetime.utcnow)
    stop_price: Optional[float] = None
    client_order_id: Optional[str] = None


@dataclass
class Trade:
    id: str
    order_id: str
    symbol: str
    side: OrderSide
    price: float
    amount: float
    fee: float
    fee_currency: str
    timestamp: datetime


@dataclass
class Balance:
    currency: str
    free: float
    used: float
    total: float


@dataclass
class Position:
    id: str
    symbol: str
    side: OrderSide
    entry_price: float
    current_price: float
    size: float
    unrealized_pnl: float
    realized_pnl: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    entry_time: datetime
    status: str = "open"


class ExchangeBase(ABC):
    def __init__(self, api_key: str, api_secret: str, sandbox: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.sandbox = sandbox
        self.name = "base"
        self.rate_limits = {}

    @abstractmethod
    async def connect(self) -> bool:
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        pass

    @abstractmethod
    async def fetch_ticker(self, symbol: str) -> Optional[Ticker]:
        pass

    @abstractmethod
    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> List[OHLCV]:
        pass

    @abstractmethod
    async def fetch_order_book(self, symbol: str, limit: int = 20) -> Optional[OrderBook]:
        pass

    @abstractmethod
    async def fetch_balance(self) -> Dict[str, Balance]:
        pass

    @abstractmethod
    async def create_order(
        self,
        symbol: str,
        order_type: OrderType,
        side: OrderSide,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict] = None
    ) -> Optional[Order]:
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        pass

    @abstractmethod
    async def fetch_order(self, order_id: str, symbol: str) -> Optional[Order]:
        pass

    @abstractmethod
    async def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        pass

    @abstractmethod
    async def fetch_my_trades(self, symbol: str, limit: int = 50) -> List[Trade]:
        pass

    @abstractmethod
    async def fetch_markets(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def fetch_usd_pairs(self) -> List[str]:
        pass

    def calculate_position_size(
        self,
        balance: float,
        risk_pct: float,
        entry_price: float,
        stop_loss: float
    ) -> float:
        risk_amount = balance * risk_pct
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit <= 0:
            return 0.0
        return risk_amount / risk_per_unit
