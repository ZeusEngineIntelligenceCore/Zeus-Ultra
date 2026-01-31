#!/usr/bin/env python3
"""
ZEUS KRAKEN EXCHANGE - Full Kraken API Integration
Secure authenticated API with order book analysis
"""

from __future__ import annotations
import asyncio
import aiohttp
import base64
import hashlib
import hmac
import time
import urllib.parse
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import logging

from .base import (
    ExchangeBase, Ticker, OHLCV, OrderBook, Balance, Order, Trade,
    OrderType, OrderSide, OrderStatus
)

logger = logging.getLogger("Zeus.Kraken")


STABLECOIN_BASES = {
    "USDT", "USDC", "DAI", "TUSD", "BUSD", "FRAX", "PYUSD",
    "USDG", "GUSD", "UST", "USTC", "EURT", "GBPQ", "USDQ"
}

TIMEFRAME_MAP = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "4h": 240, "1d": 1440, "1w": 10080
}


def normalize_asset(asset: str) -> str:
    if not asset:
        return ""
    for prefix in ("X", "Z"):
        if asset.startswith(prefix) and len(asset) > 3:
            asset = asset[1:]
    return asset


class KrakenExchange(ExchangeBase):
    BASE_URL = "https://api.kraken.com"

    def __init__(self, api_key: str = "", api_secret: str = "", sandbox: bool = False):
        super().__init__(api_key, api_secret, sandbox)
        self.name = "kraken"
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiter = asyncio.Semaphore(5)
        self._markets_cache: Dict[str, Any] = {}
        self._last_nonce = 0

    async def connect(self) -> bool:
        try:
            self.session = aiohttp.ClientSession(
                headers={"User-Agent": "ZeusBot/1.0"},
                timeout=aiohttp.ClientTimeout(total=30)
            )
            result = await self._public_request("/0/public/Time")
            if result:
                logger.info("Connected to Kraken API")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Kraken: {e}")
            return False

    async def disconnect(self) -> None:
        if self.session:
            await self.session.close()
            self.session = None

    def _get_nonce(self) -> int:
        nonce = int(time.time() * 1000)
        if nonce <= self._last_nonce:
            nonce = self._last_nonce + 1
        self._last_nonce = nonce
        return nonce

    def _sign_request(self, path: str, data: Dict[str, Any]) -> str:
        postdata = urllib.parse.urlencode(data)
        encoded = (str(data["nonce"]) + postdata).encode()
        message = path.encode() + hashlib.sha256(encoded).digest()
        signature = hmac.new(base64.b64decode(self.api_secret), message, hashlib.sha512)
        return base64.b64encode(signature.digest()).decode()

    async def _public_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Any]:
        if not self.session:
            raise RuntimeError("Session not initialized")
        async with self.rate_limiter:
            try:
                url = f"{self.BASE_URL}{endpoint}"
                async with self.session.get(url, params=params) as resp:
                    if resp.status != 200:
                        logger.error(f"HTTP {resp.status}: {await resp.text()}")
                        return None
                    data = await resp.json()
                    if data.get("error"):
                        logger.error(f"Kraken API Error: {data['error']}")
                        return None
                    return data.get("result")
            except Exception as e:
                logger.error(f"Request failed: {e}")
                return None

    async def _private_request(self, endpoint: str, data: Optional[Dict] = None) -> Optional[Any]:
        if not self.session:
            raise RuntimeError("Session not initialized")
        if not self.api_key or not self.api_secret:
            raise RuntimeError("API credentials required for private endpoints")
        data = data or {}
        data["nonce"] = self._get_nonce()
        async with self.rate_limiter:
            try:
                url = f"{self.BASE_URL}{endpoint}"
                headers = {
                    "API-Key": self.api_key,
                    "API-Sign": self._sign_request(endpoint, data),
                    "Content-Type": "application/x-www-form-urlencoded"
                }
                async with self.session.post(url, data=data, headers=headers) as resp:
                    if resp.status != 200:
                        logger.error(f"HTTP {resp.status}: {await resp.text()}")
                        return None
                    result = await resp.json()
                    if result.get("error"):
                        logger.error(f"Kraken API Error: {result['error']}")
                        return None
                    return result.get("result")
            except Exception as e:
                logger.error(f"Private request failed: {e}")
                return None

    async def fetch_ticker(self, symbol: str) -> Optional[Ticker]:
        result = await self._public_request("/0/public/Ticker", {"pair": symbol})
        if not result:
            return None
        key = next(iter(result))
        t = result[key]
        return Ticker(
            symbol=symbol,
            bid=float(t["b"][0]),
            ask=float(t["a"][0]),
            last=float(t["c"][0]),
            volume_24h=float(t["v"][1]),
            high_24h=float(t["h"][1]),
            low_24h=float(t["l"][1]),
            timestamp=datetime.now(timezone.utc)
        )

    async def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> List[OHLCV]:
        interval = TIMEFRAME_MAP.get(timeframe, 60)
        result = await self._public_request("/0/public/OHLC", {
            "pair": symbol,
            "interval": interval
        })
        if not result:
            return []
        key = next((k for k in result.keys() if k != "last"), None)
        if not key:
            return []
        candles = []
        for c in result[key][-limit:]:
            candles.append(OHLCV(
                timestamp=datetime.fromtimestamp(float(c[0]), tz=timezone.utc),
                open=float(c[1]),
                high=float(c[2]),
                low=float(c[3]),
                close=float(c[4]),
                volume=float(c[6])
            ))
        return candles

    async def fetch_order_book(self, symbol: str, limit: int = 20) -> Optional[OrderBook]:
        result = await self._public_request("/0/public/Depth", {
            "pair": symbol,
            "count": limit
        })
        if not result:
            return None
        key = next(iter(result))
        book = result[key]
        bids = [(float(p), float(v)) for p, v, _ in book.get("bids", [])]
        asks = [(float(p), float(v)) for p, v, _ in book.get("asks", [])]
        return OrderBook(
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp=datetime.now(timezone.utc)
        )

    async def fetch_balance(self) -> Dict[str, Balance]:
        result = await self._private_request("/0/private/Balance")
        if not result:
            return {}
        balances = {}
        for currency, amount in result.items():
            normalized = normalize_asset(currency)
            balances[normalized] = Balance(
                currency=normalized,
                free=float(amount),
                used=0.0,
                total=float(amount)
            )
        return balances

    async def fetch_trade_balance(self) -> Dict[str, float]:
        result = await self._private_request("/0/private/TradeBalance")
        if not result:
            return {}
        return {
            "equity": float(result.get("e", 0)),
            "trade_balance": float(result.get("tb", 0)),
            "margin": float(result.get("m", 0)),
            "unrealized_pnl": float(result.get("n", 0)),
            "free_margin": float(result.get("mf", 0))
        }

    async def create_order(
        self,
        symbol: str,
        order_type: OrderType,
        side: OrderSide,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict] = None
    ) -> Optional[Order]:
        data = {
            "pair": symbol,
            "type": side.value,
            "ordertype": order_type.value,
            "volume": f"{amount:.8f}"
        }
        if order_type == OrderType.LIMIT and price is not None:
            data["price"] = str(price)
        if params:
            data.update(params)
        result = await self._private_request("/0/private/AddOrder", data)
        if not result:
            return None
        txids = result.get("txid", [])
        if not txids:
            return None
        return Order(
            id=txids[0],
            symbol=symbol,
            side=side,
            order_type=order_type,
            price=price,
            amount=amount,
            status=OrderStatus.OPEN,
            timestamp=datetime.now(timezone.utc)
        )

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        result = await self._private_request("/0/private/CancelOrder", {"txid": order_id})
        return result is not None

    async def fetch_order(self, order_id: str, symbol: str) -> Optional[Order]:
        result = await self._private_request("/0/private/QueryOrders", {"txid": order_id})
        if not result or order_id not in result:
            return None
        o = result[order_id]
        descr = o.get("descr", {})
        return Order(
            id=order_id,
            symbol=descr.get("pair", symbol),
            side=OrderSide(descr.get("type", "buy")),
            order_type=OrderType(descr.get("ordertype", "market")),
            price=float(descr.get("price", 0)) or None,
            amount=float(o.get("vol", 0)),
            filled=float(o.get("vol_exec", 0)),
            status=self._parse_order_status(o.get("status", "")),
            timestamp=datetime.fromtimestamp(float(o.get("opentm", 0)), tz=timezone.utc)
        )

    def _parse_order_status(self, status: str) -> OrderStatus:
        status_map = {
            "pending": OrderStatus.PENDING,
            "open": OrderStatus.OPEN,
            "closed": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELED,
            "expired": OrderStatus.EXPIRED
        }
        return status_map.get(status, OrderStatus.PENDING)

    async def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        result = await self._private_request("/0/private/OpenOrders")
        if not result:
            return []
        orders = []
        for oid, o in result.get("open", {}).items():
            descr = o.get("descr", {})
            pair = descr.get("pair", "")
            if symbol and pair != symbol:
                continue
            orders.append(Order(
                id=oid,
                symbol=pair,
                side=OrderSide(descr.get("type", "buy")),
                order_type=OrderType(descr.get("ordertype", "market")),
                price=float(descr.get("price", 0)) or None,
                amount=float(o.get("vol", 0)),
                filled=float(o.get("vol_exec", 0)),
                status=OrderStatus.OPEN,
                timestamp=datetime.fromtimestamp(float(o.get("opentm", 0)), tz=timezone.utc)
            ))
        return orders

    async def fetch_my_trades(self, symbol: str, limit: int = 50) -> List[Trade]:
        result = await self._private_request("/0/private/TradesHistory")
        if not result:
            return []
        trades = []
        for tid, t in result.get("trades", {}).items():
            if t.get("pair") != symbol:
                continue
            trades.append(Trade(
                id=tid,
                order_id=t.get("ordertxid", ""),
                symbol=t.get("pair", ""),
                side=OrderSide(t.get("type", "buy")),
                price=float(t.get("price", 0)),
                amount=float(t.get("vol", 0)),
                fee=float(t.get("fee", 0)),
                fee_currency="USD",
                timestamp=datetime.fromtimestamp(float(t.get("time", 0)), tz=timezone.utc)
            ))
        return trades[:limit]

    async def fetch_markets(self) -> Dict[str, Any]:
        if self._markets_cache:
            return self._markets_cache
        result = await self._public_request("/0/public/AssetPairs")
        if result:
            self._markets_cache = result
        return self._markets_cache

    async def fetch_usd_pairs(self) -> List[str]:
        markets = await self.fetch_markets()
        pairs = []
        for name, meta in markets.items():
            quote = normalize_asset(meta.get("quote", ""))
            base = normalize_asset(meta.get("base", ""))
            if quote != "USD":
                continue
            if base in STABLECOIN_BASES:
                continue
            if ".d" in name or ".s" in name:
                continue
            pairs.append(name)
        return sorted(set(pairs))

    async def analyze_order_book(self, symbol: str) -> Dict[str, Any]:
        book = await self.fetch_order_book(symbol, 25)
        if not book:
            return {}
        return {
            "best_bid": book.best_bid(),
            "best_ask": book.best_ask(),
            "spread": book.spread(),
            "spread_pct": book.spread_pct(),
            "bid_depth": book.bid_depth(10),
            "ask_depth": book.ask_depth(10),
            "imbalance": book.imbalance(10),
            "optimal_buy": book.best_ask() - (book.spread() * 0.3),
            "optimal_sell": book.best_bid() + (book.spread() * 0.3)
        }

    async def get_optimal_entry_price(self, symbol: str, side: OrderSide) -> float:
        analysis = await self.analyze_order_book(symbol)
        if not analysis:
            ticker = await self.fetch_ticker(symbol)
            if not ticker:
                return 0.0
            return ticker.ask if side == OrderSide.BUY else ticker.bid
        if side == OrderSide.BUY:
            return analysis["optimal_buy"]
        return analysis["optimal_sell"]
