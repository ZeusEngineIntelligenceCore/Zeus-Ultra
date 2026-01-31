#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
APOLLO ZEUS: INTELLIGENCE CORE v9.0
- Scans ALL Kraken USD quote pairs (excluding stablecoin bases)
- Async OHLC + Ticker fetch
- Signal scoring + candidate ranking
- Optional LIVE execution (market entry)
- Bot-managed Stop/TP exits
- Max positions enforcement
- Kill switch
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import sys
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple

# Try to import aiohttp
try:
    import aiohttp
except ImportError:
    print("CRITICAL: 'aiohttp' is required. Please install via pip or StaSh.")
    sys.exit(1)

# ============================================================
# 1) CONFIGURATION
# ============================================================

RUN_MODE = "PAPER"  # <-- CHANGE TO "LIVE" ONLY WHEN READY
LOG_PATH = "apollo_zeus.log"
KRAKEN_API_URL = "https://api.kraken.com"

MAX_RISK_PER_TRADE = 0.01
MAX_OPEN_POSITIONS = 10
MIN_CONFIDENCE_THRESHOLD = 60.0

# Kill switch: create a file named "KILL_SWITCH" in same folder to stop trading loop
KILL_SWITCH_FILE = "KILL_SWITCH"


def kill_switch_active() -> bool:
    return os.path.exists(KILL_SWITCH_FILE)


# ============================================================
# 2) LOGGING
# ============================================================

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log = {
            "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "lvl": record.levelname,
            "mod": record.module,
            "msg": record.getMessage(),
        }
        return json.dumps(log)


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("ApolloZeus")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(JSONFormatter())
        logger.addHandler(ch)

        fh = RotatingFileHandler(LOG_PATH, maxBytes=5 * 1024 * 1024, backupCount=3)
        fh.setFormatter(JSONFormatter())
        logger.addHandler(fh)

    return logger


logger = setup_logger()

# ============================================================
# 3) STABLECOIN FILTERING + PAIR DISCOVERY HELPERS
# ============================================================

STABLECOIN_BASES = {
    "USDT", "USDC", "DAI", "TUSD", "BUSD", "FRAX", "PYUSD",
    "USDG", "GUSD", "UST", "USTC", "EURT"
}


def _normalize_asset(asset: str) -> str:
    # Kraken assets often come like: XXBT, XETH, ZUSD
    if not asset:
        return ""
    for prefix in ("X", "Z"):
        if asset.startswith(prefix) and len(asset) > 3:
            asset = asset[1:]
    return asset


# ============================================================
# 4) MATH KERNEL (PURE PYTHON)
# ============================================================

class MathKernel:
    EPS = 1e-12

    @staticmethod
    def sma(vals: List[float], length: int) -> float:
        n = min(length, len(vals))
        return sum(vals[-n:]) / float(n) if n > 0 else 0.0

    @staticmethod
    def ema(vals: List[float], length: int) -> float:
        n = min(length, len(vals))
        if n <= 1:
            return vals[-1] if vals else 0.0
        k = 2.0 / (length + 1.0)
        e = vals[-n]
        for v in vals[-n + 1:]:
            e = e + k * (v - e)
        return e

    @staticmethod
    def tr_series(high: List[float], low: List[float], close: List[float]) -> List[float]:
        n = min(len(high), len(low), len(close))
        out = []
        for i in range(1, n):
            tr = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
            out.append(tr)
        return out

    @classmethod
    def atr(cls, high: List[float], low: List[float], close: List[float], length: int = 14) -> float:
        trs = cls.tr_series(high, low, close)
        if not trs:
            return 0.0
        return cls.ema(trs, length)

    @classmethod
    def rsi(cls, close: List[float], length: int = 14) -> float:
        n = len(close)
        if n < 2:
            return 50.0

        gains, losses = [], []
        start = max(1, n - length - 5)
        for i in range(start, n):
            ch = close[i] - close[i - 1]
            gains.append(max(ch, 0.0))
            losses.append(max(-ch, 0.0))

        avg_gain = cls.ema(gains, length)
        avg_loss = cls.ema(losses, length)
        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @classmethod
    def w8_breakout_energy(
        cls,
        high: List[float],
        low: List[float],
        close: List[float],
        volume: List[float],
        win: int = 20
    ) -> float:
        if len(close) < win + 2:
            return 0.0

        curr_atr = cls.atr(high, low, close, win)
        trs = cls.tr_series(high, low, close)
        atr_base = cls.sma(trs, win) + cls.EPS
        atr_ratio = curr_atr / atr_base

        vol_ratio = volume[-1] / (cls.sma(volume, win) + cls.EPS)
        trend = 1.0 if close[-1] > cls.ema(close, win) else 0.5

        bei = atr_ratio * vol_ratio * trend
        return max(0.0, min(100.0, 20.0 * bei))

    @classmethod
    def w4_ttm_squeeze(cls, high: List[float], low: List[float], close: List[float]) -> Tuple[bool, float]:
        if len(close) < 22:
            return False, 0.0

        mid = cls.sma(close, 20)
        sl = close[-20:]
        var = sum((x - mid) ** 2 for x in sl) / 19.0
        std = var ** 0.5
        bb_width = (4.0 * std)

        curr_atr = cls.atr(high, low, close, 20)
        kc_width = (3.0 * curr_atr)

        squeeze_on = bb_width < kc_width
        return squeeze_on, (100.0 if squeeze_on else 0.0)


# ============================================================
# 5) DATA STRUCTURES + STATE
# ============================================================

@dataclass
class MarketData:
    pair: str
    price: float
    bid: float
    ask: float
    ohlc: List[Dict[str, float]]


@dataclass
class TradeSignal:
    pair: str
    action: str
    confidence: float
    stop_loss: float
    take_profit: float
    atr: float
    reasons: List[str]


@dataclass
class TradeState:
    id: str
    pair: str
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    status: str
    entry_time: float


class CoreState:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.active_trades: Dict[str, TradeState] = {}
        self.equity: float = 0.0
        self.candidates: List[Dict[str, Any]] = []
        self.health: Dict[str, Any] = {"status": "STARTING", "errors": 0}


state = CoreState()

# ============================================================
# 6) KRAKEN FEED (ASYNC + PRIVATE ENDPOINTS)
# ============================================================

class KrakenFeed:
    def __init__(self, key: str, secret: str):
        self.key = (key or "").strip()
        self.secret = (secret or "").strip()
        self.limiter = asyncio.Semaphore(5)
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(headers={"User-Agent": "ApolloZeus/9.0"})
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _sign(self, path: str, data: Dict[str, Any]) -> str:
        postdata = urllib.parse.urlencode(data)
        encoded = (str(data["nonce"]) + postdata).encode()
        message = path.encode() + hashlib.sha256(encoded).digest()
        signature = hmac.new(base64.b64decode(self.secret), message, hashlib.sha512)
        return base64.b64encode(signature.digest()).decode()

    async def _request(self, method: str, uri: str, public: bool = True, data: Optional[Dict[str, Any]] = None) -> Any:
        if not self.session:
            raise RuntimeError("KrakenFeed session not initialized. Use 'async with KrakenFeed(...)'.")

        url = KRAKEN_API_URL + uri
        data = data or {}

        async with self.limiter:
            if not public:
                data["nonce"] = int(time.time() * 1000)
                headers = {
                    "API-Key": self.key,
                    "API-Sign": self._sign(uri, data),
                    "Content-Type": "application/x-www-form-urlencoded",
                }
                async with self.session.post(url, data=data, headers=headers) as resp:
                    return await self._process_resp(resp)
            else:
                async with self.session.get(url, params=data) as resp:
                    return await self._process_resp(resp)

    async def _process_resp(self, resp) -> Any:
        if resp.status != 200:
            logger.error(f"HTTP Error {resp.status}")
            return None

        try:
            js = await resp.json()
        except Exception:
            return None

        if js.get("error"):
            err = js["error"]
            logger.error(f"Kraken API Error: {err}")
            return None

        return js.get("result")

    async def fetch_ohlc(self, pair: str, interval: int = 60) -> Optional[MarketData]:
        tasks = [
            self._request("GET", "/0/public/OHLC", public=True, data={"pair": pair, "interval": interval}),
            self._request("GET", "/0/public/Ticker", public=True, data={"pair": pair}),
        ]
        ohlc_data, ticker_data = await asyncio.gather(*tasks)

        if not ohlc_data or not ticker_data:
            return None

        data_key = next((k for k in ohlc_data.keys() if isinstance(ohlc_data[k], list)), None)
        if not data_key:
            return None

        candles = []
        for c in ohlc_data[data_key][-60:]:
            candles.append({
                "time": float(c[0]),
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[6]),
            })

        t_key = next(iter(ticker_data))
        t = ticker_data[t_key]

        return MarketData(
            pair=pair,
            price=float(t["c"][0]),
            bid=float(t["b"][0]),
            ask=float(t["a"][0]),
            ohlc=candles,
        )

    async def get_balance(self) -> float:
        res = await self._request("POST", "/0/private/Balance", public=False, data={})
        if not res:
            return 0.0
        return float(res.get("ZUSD", res.get("USD", 0.0)))

    # -------- PRIVATE ORDER ENDPOINTS --------

    async def add_order(
        self,
        pair: str,
        side: str,
        volume: float,
        ordertype: str = "market",
        price: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        data = {
            "pair": pair,
            "type": side.lower(),
            "ordertype": ordertype,
            "volume": f"{volume:.8f}",
        }
        if ordertype == "limit" and price is not None:
            data["price"] = str(price)

        return await self._request("POST", "/0/private/AddOrder", public=False, data=data)

    async def open_orders(self) -> Optional[Dict[str, Any]]:
        return await self._request("POST", "/0/private/OpenOrders", public=False, data={})

    async def query_orders(self, txids: str) -> Optional[Dict[str, Any]]:
        return await self._request("POST", "/0/private/QueryOrders", public=False, data={"txid": txids})

    async def cancel_order(self, txid: str) -> Optional[Dict[str, Any]]:
        return await self._request("POST", "/0/private/CancelOrder", public=False, data={"txid": txid})


# ============================================================
# 7) PAIR DISCOVERY (NOW THAT KrakenFeed EXISTS)
# ============================================================

async def discover_usd_pairs(feed: KrakenFeed) -> List[str]:
    res = await feed._request("GET", "/0/public/AssetPairs", public=True, data={})
    if not res:
        return []

    pairs = []
    for name, meta in res.items():
        quote = _normalize_asset(meta.get("quote", ""))
        base = _normalize_asset(meta.get("base", ""))

        if quote != "USD":
            continue
        if base in STABLECOIN_BASES:
            continue
        if ".d" in name or ".s" in name:
            continue

        pairs.append(name)

    return sorted(set(pairs))


# ============================================================
# 8) ALPHA + RISK
# ============================================================

class AlphaEngine:
    @staticmethod
    def analyze(md: MarketData) -> Optional[TradeSignal]:
        closes = [c["close"] for c in md.ohlc]
        highs = [c["high"] for c in md.ohlc]
        lows = [c["low"] for c in md.ohlc]
        vols = [c["volume"] for c in md.ohlc]

        if len(closes) < 30:
            return None

        atr_val = MathKernel.atr(highs, lows, closes, 14)
        if atr_val == 0:
            return None

        score_breakout = MathKernel.w8_breakout_energy(highs, lows, closes, vols)
        is_squeeze, _ = MathKernel.w4_ttm_squeeze(highs, lows, closes)
        rsi = MathKernel.rsi(closes, 14)

        reasons = []
        confidence = 0.0

        if score_breakout > 50 and rsi > 55:
            confidence += 40
            reasons.append(f"Breakout Energy: {score_breakout:.1f}")

        if (not is_squeeze) and score_breakout > 20:
            confidence += 20
            reasons.append("Volatility Squeeze Release")

        if 50 < rsi < 75:
            confidence += 15
            reasons.append(f"Bullish RSI {rsi:.1f}")

        stop_price = md.price - (2.0 * atr_val)
        target_price = md.price + (3.0 * atr_val)

        if confidence >= MIN_CONFIDENCE_THRESHOLD:
            return TradeSignal(
                pair=md.pair,
                action="BUY",
                confidence=confidence,
                stop_loss=stop_price,
                take_profit=target_price,
                atr=atr_val,
                reasons=reasons
            )
        return None


class RiskManager:
    @staticmethod
    def size_position(signal: TradeSignal, equity: float, entry_price: float) -> float:
        risk_amt = equity * MAX_RISK_PER_TRADE
        risk_per_unit = abs(entry_price - signal.stop_loss)
        if risk_per_unit <= 0:
            return 0.0
        return risk_amt / risk_per_unit


# ============================================================
# 9) EXECUTION HELPERS
# ============================================================

async def can_open_new_trade() -> bool:
    async with state.lock:
        open_count = sum(1 for t in state.active_trades.values() if t.status in ("OPEN", "FILLED"))
    return open_count < MAX_OPEN_POSITIONS


def new_trade_id() -> str:
    return f"T{int(time.time() * 1000)}"


async def register_trade(signal: TradeSignal, entry_price: float, size: float, txid: str):
    ts = TradeState(
        id=new_trade_id(),
        pair=signal.pair,
        entry_price=entry_price,
        size=size,
        stop_loss=signal.stop_loss,
        take_profit=signal.take_profit,
        status="OPEN",
        entry_time=time.time(),
    )
    async with state.lock:
        state.active_trades[txid] = ts


async def manage_exits(feed: KrakenFeed):
    async with state.lock:
        trades = list(state.active_trades.items())

    for txid, t in trades:
        if t.status not in ("OPEN", "FILLED"):
            continue

        md = await feed.fetch_ohlc(t.pair)
        if not md:
            continue

        px = md.price

        if px <= t.stop_loss:
            logger.warning(f"STOP HIT {t.pair} @ {px}")
            if RUN_MODE == "LIVE":
                await feed.add_order(pair=t.pair, side="sell", volume=t.size, ordertype="market")
            async with state.lock:
                state.active_trades[txid].status = "CLOSED"
            continue

        if px >= t.take_profit:
            logger.warning(f"TP HIT {t.pair} @ {px}")
            if RUN_MODE == "LIVE":
                await feed.add_order(pair=t.pair, side="sell", volume=t.size, ordertype="market")
            async with state.lock:
                state.active_trades[txid].status = "CLOSED"
            continue


async def maybe_execute_top_candidate(feed: KrakenFeed):
    if RUN_MODE != "LIVE":
        return

    if not await can_open_new_trade():
        logger.warning("Max open positions reached. Skipping execution.")
        return

    async with state.lock:
        if not state.candidates:
            return
        top = state.candidates[0]

    pair = top["pair"]
    size = float(top["size"])
    price = float(top["price"])

    if size <= 0:
        return

    res = await feed.add_order(pair=pair, side="buy", volume=size, ordertype="market")
    if not res:
        logger.error("Order placement failed.")
        return

    txid = None
    txids = res.get("txid") if isinstance(res, dict) else None
    if isinstance(txids, list) and txids:
        txid = txids[0]

    if not txid:
        logger.error(f"Order returned no txid: {res}")
        return

    logger.info(f"ORDER PLACED {pair} size={size} txid={txid}")

    fake_signal = TradeSignal(
        pair=pair,
        action="BUY",
        confidence=top["signal"],
        stop_loss=top["stop"],
        take_profit=top["target"],
        atr=0.0,
        reasons=top["reasons"],
    )
    await register_trade(fake_signal, entry_price=price, size=size, txid=txid)


# ============================================================
# 10) MARKET CYCLE
# ============================================================

async def run_market_cycle(feed, pairs: List[str]) -> None:
    logger.info("Starting Market Cycle...", extra={"pairs": len(pairs)})

    # ---- Snapshot equity + mark scanning ----
    async with state.lock:
        state.health["status"] = "SCANNING"
        current_equity = float(state.equity or 0.0)

    # ---- Batch fetch to avoid 600+ task backlog ----
    BATCH_SIZE = 40  # 25-60 is a good range for Pythonista
    results: List[Optional[MarketData]] = []

    total = len(pairs)
    scanned = 0

    for i in range(0, total, BATCH_SIZE):
        batch = pairs[i:i + BATCH_SIZE]
        batch_tasks = [feed.fetch_ohlc(p) for p in batch]

        try:
            batch_results = await asyncio.gather(*batch_tasks)
        except Exception:
            logger.exception("Batch fetch failed")
            batch_results = []

        results.extend(batch_results)
        scanned += len(batch)

        # heartbeat progress (no spam)
        if scanned % (BATCH_SIZE * 5) == 0 or scanned >= total:
            logger.info(f"Progress: {scanned}/{total} scanned")

    # ---- Analyze ----
    opportunities: List[Dict[str, Any]] = []

    for md in results:
        if not md:
            continue

        signal = AlphaEngine.analyze(md)
        if not signal:
            continue

        units = RiskManager.size_position(signal, current_equity, md.price)
        if units <= 0:
            continue

        notional = units * md.price

        opp = {
            "pair": signal.pair,
            "signal": float(signal.confidence),
            "price": float(md.price),
            "stop": float(signal.stop_loss),
            "target": float(signal.take_profit),
            "size": float(units),
            "notional_usd": float(notional),
            "reasons": list(signal.reasons),
        }
        opportunities.append(opp)

        # log only when a real signal happens
        logger.info(
            f"SIGNAL {signal.pair}",
            extra={
                "confidence": float(signal.confidence),
                "price": float(md.price),
                "stop": float(signal.stop_loss),
                "target": float(signal.take_profit),
            },
        )

    # ---- Update shared state ONCE ----
    candidates_sorted = sorted(opportunities, key=lambda x: x["signal"], reverse=True)

    async with state.lock:
        state.candidates = candidates_sorted
        state.health["status"] = "IDLE"

    # ---- Terminal Output (top 5) ----
    print("\n" + "=" * 50)
    print(f"APOLLO ZEUS v9 | {datetime.now().strftime('%H:%M:%S')} | Mode: {RUN_MODE}")
    print(f"Equity: ${current_equity:.2f} | Active Candidates: {len(candidates_sorted)}")
    print("-" * 50)

    for c in candidates_sorted[:5]:
        denom = (c["price"] - c["stop"])
        rr = (c["target"] - c["price"]) / denom if denom != 0 else float("inf")

        print(f"  > {c['pair']:<10} | Score: {c['signal']:.1f} | R:R {rr:.2f}")
        print(f"    Price: {c['price']:.6f} | Stop: {c['stop']:.6f} | Target: {c['target']:.6f}")
        print(f"    Size: {c['size']:.8f} | Notional: ${c['notional_usd']:.2f}")
        print(f"    Reasons: {', '.join(c['reasons'])}")
        print("=" * 50 + "\n")


# ============================================================
# 11) MAIN
# ============================================================

async def main():
    # ---- Load secrets ----
    search_root = os.path.dirname(os.path.abspath(__file__))
    target_file = "secrets.json"
    secrets_path = None

    for root, dirs, files in os.walk(search_root):
        if target_file in files:
            secrets_path = os.path.join(root, target_file)
            break

    if not secrets_path:
        doc_path = os.path.expanduser("~/Documents")
        for root, dirs, files in os.walk(doc_path):
            if target_file in files:
                secrets_path = os.path.join(root, target_file)
                break

    if not secrets_path:
        print(f"❌ FATAL: Could not find '{target_file}'")
        return

    with open(secrets_path, "r") as f:
        secrets = json.load(f)

    api_key = (secrets.get("kraken_api_key") or "").strip()
    api_secret = (secrets.get("kraken_api_secret") or "").strip()

    print(f"✅ Loaded secrets from: {secrets_path}")

    async with state.lock:
        state.equity = 10000.0  # PAPER default; replace with feed.get_balance() if LIVE

    async with KrakenFeed(api_key, api_secret) as feed:
        target_pairs = await discover_usd_pairs(feed)
        print(f"🧠 Discovered {len(target_pairs)} USD pairs (stablecoin bases removed).")

        while True:
            try:
                if kill_switch_active():
                    logger.error("KILL SWITCH ACTIVE - trading halted")
                    async with state.lock:
                        state.health["status"] = "KILLED"
                    await asyncio.sleep(5)
                    continue

                await run_market_cycle(feed, target_pairs)

                # Execute best trade (LIVE only)
                await maybe_execute_top_candidate(feed)

                # Manage exits (stop/tp)
                await manage_exits(feed)

            except Exception:
                logger.exception("Critical Loop Failure")

            await asyncio.sleep(10)


if __name__ == "__main__":
    asyncio.run(main())
