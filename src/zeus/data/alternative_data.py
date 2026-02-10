#!/usr/bin/env python3
"""
ZEUS ALTERNATIVE DATA INTEGRATION
Institutional-grade alternative data sources:
- Fear & Greed Index
- Funding Rates (perpetual futures)
- Open Interest Analysis
- Whale Wallet Tracking
- Social Sentiment Analysis
"""

from __future__ import annotations
import asyncio
import aiohttp
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger("Zeus.AltData")


@dataclass
class FearGreedData:
    value: int
    classification: str
    timestamp: datetime
    previous_value: int = 0
    trend: str = "stable"


@dataclass
class FundingRateData:
    symbol: str
    rate: float
    predicted_rate: float
    timestamp: datetime
    is_positive: bool
    annualized_rate: float


@dataclass
class OpenInterestData:
    symbol: str
    open_interest: float
    oi_change_24h: float
    oi_change_pct: float
    long_short_ratio: float
    timestamp: datetime


@dataclass
class WhaleActivity:
    symbol: str
    transaction_type: str
    amount: float
    usd_value: float
    from_address: str
    to_address: str
    timestamp: datetime
    is_exchange_flow: bool


@dataclass
class SentimentData:
    symbol: str
    bullish_pct: float
    bearish_pct: float
    neutral_pct: float
    social_volume: int
    sentiment_score: float
    timestamp: datetime


@dataclass
class AlternativeDataSignal:
    fear_greed: Optional[FearGreedData]
    funding_rates: Dict[str, FundingRateData]
    open_interest: Dict[str, OpenInterestData]
    whale_activities: List[WhaleActivity]
    sentiment: Optional[SentimentData]
    macro_score: float
    signals: List[str] = field(default_factory=list)
    last_update: datetime = field(default_factory=datetime.now)


class FearGreedIndexFetcher:
    API_URL = "https://api.alternative.me/fng/"
    
    def __init__(self):
        self.cache: Optional[FearGreedData] = None
        self.cache_time: Optional[datetime] = None
        self.cache_duration = timedelta(minutes=30)
    
    async def fetch(self) -> Optional[FearGreedData]:
        if self.cache and self.cache_time:
            if datetime.now() - self.cache_time < self.cache_duration:
                return self.cache
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.API_URL, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get("data"):
                            current = data["data"][0]
                            previous = data["data"][1] if len(data["data"]) > 1 else current
                            
                            value = int(current.get("value", 50))
                            prev_value = int(previous.get("value", value))
                            
                            if value > prev_value + 5:
                                trend = "improving"
                            elif value < prev_value - 5:
                                trend = "declining"
                            else:
                                trend = "stable"
                            
                            self.cache = FearGreedData(
                                value=value,
                                classification=current.get("value_classification", "Neutral"),
                                timestamp=datetime.now(),
                                previous_value=prev_value,
                                trend=trend
                            )
                            self.cache_time = datetime.now()
                            
                            return self.cache
        except Exception as e:
            logger.debug(f"Fear & Greed fetch error: {e}")
        
        return self.cache


class FundingRateTracker:
    def __init__(self):
        self.rates: Dict[str, FundingRateData] = {}
        self.history: Dict[str, List[float]] = {}
    
    def update(self, symbol: str, rate: float, predicted_rate: float = 0.0) -> FundingRateData:
        if symbol not in self.history:
            self.history[symbol] = []
        
        self.history[symbol].append(rate)
        if len(self.history[symbol]) > 100:
            self.history[symbol] = self.history[symbol][-100:]
        
        annualized = rate * 3 * 365
        
        data = FundingRateData(
            symbol=symbol,
            rate=rate,
            predicted_rate=predicted_rate,
            timestamp=datetime.now(),
            is_positive=rate > 0,
            annualized_rate=annualized
        )
        
        self.rates[symbol] = data
        return data
    
    def get_funding_signal(self, symbol: str) -> Dict[str, Any]:
        if symbol not in self.rates:
            return {"signal": "neutral", "score": 0}
        
        data = self.rates[symbol]
        
        if data.rate > 0.001:
            return {
                "signal": "bearish",
                "reason": "High positive funding - longs paying shorts",
                "score": -min(20, data.rate * 10000)
            }
        elif data.rate < -0.001:
            return {
                "signal": "bullish",
                "reason": "Negative funding - shorts paying longs",
                "score": min(20, abs(data.rate) * 10000)
            }
        
        return {"signal": "neutral", "score": 0}


class OpenInterestTracker:
    def __init__(self):
        self.data: Dict[str, OpenInterestData] = {}
        self.history: Dict[str, List[float]] = {}
    
    def update(self, symbol: str, oi: float, long_short_ratio: float = 1.0) -> OpenInterestData:
        prev_oi = self.history.get(symbol, [oi])[-1] if symbol in self.history else oi
        
        if symbol not in self.history:
            self.history[symbol] = []
        
        self.history[symbol].append(oi)
        if len(self.history[symbol]) > 100:
            self.history[symbol] = self.history[symbol][-100:]
        
        oi_24h_ago = self.history[symbol][0] if len(self.history[symbol]) >= 24 else oi
        oi_change_24h = oi - oi_24h_ago
        oi_change_pct = (oi_change_24h / oi_24h_ago) if oi_24h_ago > 0 else 0
        
        data = OpenInterestData(
            symbol=symbol,
            open_interest=oi,
            oi_change_24h=oi_change_24h,
            oi_change_pct=oi_change_pct,
            long_short_ratio=long_short_ratio,
            timestamp=datetime.now()
        )
        
        self.data[symbol] = data
        return data
    
    def get_oi_signal(self, symbol: str, price_direction: str) -> Dict[str, Any]:
        if symbol not in self.data:
            return {"signal": "neutral", "score": 0}
        
        data = self.data[symbol]
        
        if data.oi_change_pct > 0.1 and price_direction == "up":
            return {
                "signal": "bullish",
                "reason": "Rising OI with rising price - new longs entering",
                "score": 15
            }
        elif data.oi_change_pct > 0.1 and price_direction == "down":
            return {
                "signal": "bearish",
                "reason": "Rising OI with falling price - new shorts entering",
                "score": -15
            }
        elif data.oi_change_pct < -0.1 and price_direction == "up":
            return {
                "signal": "weak_bullish",
                "reason": "Falling OI with rising price - short squeeze",
                "score": 5
            }
        elif data.oi_change_pct < -0.1 and price_direction == "down":
            return {
                "signal": "weak_bearish",
                "reason": "Falling OI with falling price - long liquidations",
                "score": -5
            }
        
        if data.long_short_ratio > 2.0:
            return {
                "signal": "contrarian_bearish",
                "reason": "Extreme long positioning - potential reversal",
                "score": -10
            }
        elif data.long_short_ratio < 0.5:
            return {
                "signal": "contrarian_bullish",
                "reason": "Extreme short positioning - potential squeeze",
                "score": 10
            }
        
        return {"signal": "neutral", "score": 0}


class WhaleTracker:
    def __init__(self, min_whale_usd: float = 100000):
        self.min_whale_usd = min_whale_usd
        self.activities: List[WhaleActivity] = []
        self.exchange_addresses: set = set()
    
    def add_activity(self, symbol: str, tx_type: str, amount: float,
                    usd_value: float, from_addr: str, to_addr: str,
                    is_exchange: bool = False) -> Optional[WhaleActivity]:
        if usd_value < self.min_whale_usd:
            return None
        
        activity = WhaleActivity(
            symbol=symbol,
            transaction_type=tx_type,
            amount=amount,
            usd_value=usd_value,
            from_address=from_addr,
            to_address=to_addr,
            timestamp=datetime.now(),
            is_exchange_flow=is_exchange
        )
        
        self.activities.append(activity)
        
        if len(self.activities) > 500:
            self.activities = self.activities[-500:]
        
        return activity
    
    def get_whale_signal(self, symbol: str, hours: int = 24) -> Dict[str, Any]:
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [a for a in self.activities 
                  if a.symbol == symbol and a.timestamp > cutoff]
        
        if not recent:
            return {"signal": "neutral", "score": 0}
        
        exchange_inflows = sum(a.usd_value for a in recent 
                               if a.is_exchange_flow and a.transaction_type == "deposit")
        exchange_outflows = sum(a.usd_value for a in recent 
                                if a.is_exchange_flow and a.transaction_type == "withdraw")
        
        net_flow = exchange_outflows - exchange_inflows
        
        if net_flow > 1000000:
            return {
                "signal": "bullish",
                "reason": f"${net_flow/1e6:.1f}M net outflow from exchanges",
                "score": min(20, net_flow / 500000)
            }
        elif net_flow < -1000000:
            return {
                "signal": "bearish",
                "reason": f"${abs(net_flow)/1e6:.1f}M net inflow to exchanges",
                "score": max(-20, net_flow / 500000)
            }
        
        return {"signal": "neutral", "score": 0}


class AlternativeDataAggregator:
    def __init__(self):
        self.fear_greed = FearGreedIndexFetcher()
        self.funding = FundingRateTracker()
        self.open_interest = OpenInterestTracker()
        self.whales = WhaleTracker()
        self.last_update: Optional[datetime] = None
        self.cached_signal: Optional[AlternativeDataSignal] = None
    
    def get_fear_greed(self) -> Optional[FearGreedData]:
        """Get cached fear/greed data synchronously (no API call)"""
        return self.fear_greed.cache
    
    async def update_fear_greed(self) -> Optional[FearGreedData]:
        return await self.fear_greed.fetch()
    
    def update_funding(self, symbol: str, rate: float, predicted: float = 0.0) -> FundingRateData:
        return self.funding.update(symbol, rate, predicted)
    
    def update_oi(self, symbol: str, oi: float, ls_ratio: float = 1.0) -> OpenInterestData:
        return self.open_interest.update(symbol, oi, ls_ratio)
    
    def add_whale_activity(self, symbol: str, tx_type: str, amount: float,
                          usd_value: float, from_addr: str, to_addr: str,
                          is_exchange: bool = False) -> Optional[WhaleActivity]:
        return self.whales.add_activity(symbol, tx_type, amount, usd_value,
                                        from_addr, to_addr, is_exchange)
    
    async def get_aggregated_signal(self, symbol: Optional[str] = None,
                                   price_direction: str = "neutral") -> AlternativeDataSignal:
        signals = []
        macro_score = 50.0
        
        fg_data = await self.fear_greed.fetch()
        
        if fg_data:
            if fg_data.value < 25:
                signals.append(f"Extreme Fear ({fg_data.value}) - Contrarian buy signal")
                macro_score += 15
            elif fg_data.value < 40:
                signals.append(f"Fear ({fg_data.value}) - Cautious accumulation zone")
                macro_score += 8
            elif fg_data.value > 75:
                signals.append(f"Extreme Greed ({fg_data.value}) - Contrarian sell signal")
                macro_score -= 15
            elif fg_data.value > 60:
                signals.append(f"Greed ({fg_data.value}) - Risk of correction")
                macro_score -= 8
            
            if fg_data.trend == "improving":
                macro_score += 5
            elif fg_data.trend == "declining":
                macro_score -= 5
        
        funding_rates = {}
        if symbol and symbol in self.funding.rates:
            funding_rates[symbol] = self.funding.rates[symbol]
            funding_signal = self.funding.get_funding_signal(symbol)
            if funding_signal["signal"] != "neutral":
                signals.append(funding_signal["reason"])
                macro_score += funding_signal["score"]
        
        oi_data = {}
        if symbol and symbol in self.open_interest.data:
            oi_data[symbol] = self.open_interest.data[symbol]
            oi_signal = self.open_interest.get_oi_signal(symbol, price_direction)
            if oi_signal["signal"] != "neutral":
                signals.append(oi_signal["reason"])
                macro_score += oi_signal["score"]
        
        whale_signal = self.whales.get_whale_signal(symbol) if symbol else {"signal": "neutral", "score": 0}
        if whale_signal["signal"] != "neutral":
            signals.append(whale_signal["reason"])
            macro_score += whale_signal["score"]
        
        macro_score = max(0, min(100, macro_score))
        
        return AlternativeDataSignal(
            fear_greed=fg_data,
            funding_rates=funding_rates,
            open_interest=oi_data,
            whale_activities=self.whales.activities[-10:] if symbol is None else 
                            [a for a in self.whales.activities[-10:] if a.symbol == symbol],
            sentiment=None,
            macro_score=macro_score,
            signals=signals,
            last_update=datetime.now()
        )
    
    def get_quick_signal(self, symbol: str, price_direction: str = "neutral") -> Dict[str, Any]:
        score = 50.0
        signals = []
        
        if self.fear_greed.cache:
            fg = self.fear_greed.cache
            if fg.value < 30:
                score += 10
                signals.append("Market fear - buying opportunity")
            elif fg.value > 70:
                score -= 10
                signals.append("Market greed - caution advised")
        
        if symbol in self.funding.rates:
            f_sig = self.funding.get_funding_signal(symbol)
            score += f_sig.get("score", 0)
            if f_sig["signal"] != "neutral":
                signals.append(f_sig["reason"])
        
        if symbol in self.open_interest.data:
            oi_sig = self.open_interest.get_oi_signal(symbol, price_direction)
            score += oi_sig.get("score", 0)
            if oi_sig["signal"] != "neutral":
                signals.append(oi_sig["reason"])
        
        w_sig = self.whales.get_whale_signal(symbol)
        score += w_sig.get("score", 0)
        if w_sig["signal"] != "neutral":
            signals.append(w_sig["reason"])
        
        return {
            "score": max(0, min(100, score)),
            "signals": signals,
            "recommendation": "bullish" if score > 60 else "bearish" if score < 40 else "neutral"
        }
