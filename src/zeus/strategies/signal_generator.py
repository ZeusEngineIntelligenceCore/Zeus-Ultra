#!/usr/bin/env python3
"""
ZEUS SIGNAL GENERATOR - Advanced Trading Signal Engine
Combines multiple indicators and timeframe analysis for high-confidence signals
"""

from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from enum import Enum
import math

from ..indicators.math_kernel import MathKernel
from ..indicators.prebreakout_detector import PreBreakoutDetector, MultiTimeframeAnalyzer
from ..exchanges.base import OHLCV, OrderSide


class SignalType(Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    WEAK_BUY = "weak_buy"
    NEUTRAL = "neutral"
    WEAK_SELL = "weak_sell"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class StrategyMode(Enum):
    SCALP = "scalp"
    DAY_TRADE = "day_trade"
    SWING = "swing"
    PRE_BREAKOUT = "pre_breakout"


@dataclass
class TradingSignal:
    symbol: str
    signal_type: SignalType
    side: OrderSide
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    take_profit_levels: List[float] = field(default_factory=list)
    atr: float = 0.0
    risk_reward: float = 0.0
    reasons: List[str] = field(default_factory=list)
    indicators: Dict[str, Any] = field(default_factory=dict)
    strategy_mode: StrategyMode = StrategyMode.DAY_TRADE
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    prebreakout_score: float = 0.0
    timeframe_alignment: float = 0.0


@dataclass
class SignalConfig:
    min_confidence: float = 65.0
    min_rr_ratio: float = 1.5
    max_spread_pct: float = 0.5
    atr_stop_mult: float = 2.0
    atr_tp_mult: float = 3.0
    scalp_atr_stop: float = 1.0
    scalp_atr_tp: float = 1.5
    swing_atr_stop: float = 2.5
    swing_atr_tp: float = 4.0
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    volume_spike_threshold: float = 1.5
    trend_ema_fast: int = 9
    trend_ema_slow: int = 21
    trend_ema_trend: int = 50


class SignalGenerator:
    def __init__(self, config: Optional[SignalConfig] = None):
        self.config = config or SignalConfig()
        self.math = MathKernel()
        self.prebreakout = PreBreakoutDetector()
        self.mtf_analyzer = MultiTimeframeAnalyzer()

    async def generate_signal(
        self,
        symbol: str,
        ohlcv: List[OHLCV],
        mode: StrategyMode = StrategyMode.DAY_TRADE,
        order_book_analysis: Optional[Dict] = None
    ) -> Optional[TradingSignal]:
        if len(ohlcv) < 50:
            return None
        high = [c.high for c in ohlcv]
        low = [c.low for c in ohlcv]
        close = [c.close for c in ohlcv]
        volume = [c.volume for c in ohlcv]
        indicators = self.math.calculate_all(high, low, close, volume)
        prebreakout = await self.prebreakout.analyze(symbol, high, low, close, volume)
        trend = self._analyze_trend(close)
        momentum = self._analyze_momentum(indicators)
        volatility = self._analyze_volatility(indicators, high, low, close)
        volume_analysis = self._analyze_volume(volume, indicators)
        signal_score, signal_reasons = self._calculate_signal_score(
            trend, momentum, volatility, volume_analysis, prebreakout, mode
        )
        if signal_score < self.config.min_confidence:
            return None
        side = OrderSide.BUY if signal_score > 0 else OrderSide.SELL
        abs_score = abs(signal_score)
        signal_type = self._determine_signal_type(abs_score, side)
        current_price = close[-1]
        atr = indicators["atr"]
        if mode == StrategyMode.SCALP:
            stop_mult = self.config.scalp_atr_stop
            tp_mult = self.config.scalp_atr_tp
        elif mode == StrategyMode.SWING:
            stop_mult = self.config.swing_atr_stop
            tp_mult = self.config.swing_atr_tp
        else:
            stop_mult = self.config.atr_stop_mult
            tp_mult = self.config.atr_tp_mult
        if side == OrderSide.BUY:
            stop_loss = current_price - (stop_mult * atr)
            take_profit = current_price + (tp_mult * atr)
            tp_levels = [
                current_price + (tp_mult * 0.5 * atr),
                current_price + (tp_mult * atr),
                current_price + (tp_mult * 1.5 * atr)
            ]
        else:
            stop_loss = current_price + (stop_mult * atr)
            take_profit = current_price - (tp_mult * atr)
            tp_levels = [
                current_price - (tp_mult * 0.5 * atr),
                current_price - (tp_mult * atr),
                current_price - (tp_mult * 1.5 * atr)
            ]
        if prebreakout.get("stage") == "PRE_BREAKOUT":
            if side == OrderSide.BUY:
                stop_loss = min(stop_loss, prebreakout.get("buy_anchor", stop_loss) - atr)
                take_profit = max(take_profit, prebreakout.get("sell_anchor", take_profit))
        risk = abs(current_price - stop_loss)
        reward = abs(take_profit - current_price)
        risk_reward = reward / risk if risk > 0 else 0
        if risk_reward < self.config.min_rr_ratio:
            return None
        return TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            side=side,
            confidence=abs_score,
            entry_price=current_price,
            stop_loss=round(stop_loss, 8),
            take_profit=round(take_profit, 8),
            take_profit_levels=[round(tp, 8) for tp in tp_levels],
            atr=round(atr, 8),
            risk_reward=round(risk_reward, 2),
            reasons=signal_reasons,
            indicators={
                "rsi": round(indicators["rsi"], 2),
                "macd": indicators["macd"],
                "adx": indicators["adx"],
                "squeeze": indicators["ttm_squeeze"],
                "breakout_energy": round(indicators["breakout_energy"], 2),
                "volume_profile": indicators["volume_profile"]
            },
            strategy_mode=mode,
            prebreakout_score=prebreakout.get("prebreakout_score", 0),
            timeframe_alignment=prebreakout.get("features", {}).get("microtrend", 0)
        )

    def _analyze_trend(self, close: List[float]) -> Dict[str, Any]:
        ema_fast = self.math.ema(close, self.config.trend_ema_fast)
        ema_slow = self.math.ema(close, self.config.trend_ema_slow)
        ema_trend = self.math.ema(close, self.config.trend_ema_trend)
        current = close[-1]
        if ema_fast > ema_slow > ema_trend and current > ema_fast:
            direction = "STRONG_UP"
            strength = 1.0
        elif ema_fast > ema_slow and current > ema_slow:
            direction = "UP"
            strength = 0.7
        elif ema_fast < ema_slow < ema_trend and current < ema_fast:
            direction = "STRONG_DOWN"
            strength = -1.0
        elif ema_fast < ema_slow and current < ema_slow:
            direction = "DOWN"
            strength = -0.7
        else:
            direction = "NEUTRAL"
            strength = 0.0
        return {
            "direction": direction,
            "strength": strength,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "ema_trend": ema_trend
        }

    def _analyze_momentum(self, indicators: Dict) -> Dict[str, Any]:
        rsi = indicators["rsi"]
        macd_line, signal_line, histogram = indicators["macd"]
        stoch_k, stoch_d = indicators["stoch_rsi"]
        adx, plus_di, minus_di = indicators["adx"]
        momentum_score = 0.0
        signals = []
        if rsi < self.config.rsi_oversold:
            momentum_score += 30
            signals.append("RSI Oversold")
        elif rsi > self.config.rsi_overbought:
            momentum_score -= 30
            signals.append("RSI Overbought")
        elif 40 < rsi < 60:
            momentum_score += 10
            signals.append("RSI Neutral Zone")
        if histogram > 0 and macd_line > signal_line:
            momentum_score += 20
            signals.append("MACD Bullish")
        elif histogram < 0 and macd_line < signal_line:
            momentum_score -= 20
            signals.append("MACD Bearish")
        if adx > 25:
            if plus_di > minus_di:
                momentum_score += 15
                signals.append("Strong Bullish Trend")
            else:
                momentum_score -= 15
                signals.append("Strong Bearish Trend")
        if stoch_k < 20:
            momentum_score += 10
            signals.append("Stochastic Oversold")
        elif stoch_k > 80:
            momentum_score -= 10
            signals.append("Stochastic Overbought")
        return {
            "score": momentum_score,
            "signals": signals,
            "rsi": rsi,
            "macd_histogram": histogram,
            "adx": adx
        }

    def _analyze_volatility(self, indicators: Dict, high: List[float], 
                           low: List[float], close: List[float]) -> Dict[str, Any]:
        squeeze_on, momentum, bb_width = indicators["ttm_squeeze"]
        atr = indicators["atr"]
        bb_upper, bb_mid, bb_lower = indicators["bollinger"]
        current = close[-1]
        volatility_score = 0.0
        signals = []
        if squeeze_on and momentum > 0:
            volatility_score += 25
            signals.append("TTM Squeeze Bullish Fire")
        elif squeeze_on and momentum < 0:
            volatility_score -= 15
            signals.append("TTM Squeeze Bearish")
        elif not squeeze_on and momentum > 0:
            volatility_score += 10
            signals.append("Squeeze Released Bullish")
        if current < bb_lower:
            volatility_score += 15
            signals.append("Price Below Lower BB")
        elif current > bb_upper:
            volatility_score -= 15
            signals.append("Price Above Upper BB")
        breakout_energy = indicators["breakout_energy"]
        if breakout_energy > 60:
            volatility_score += 20
            signals.append(f"High Breakout Energy ({breakout_energy:.1f})")
        return {
            "score": volatility_score,
            "signals": signals,
            "squeeze_on": squeeze_on,
            "atr": atr,
            "breakout_energy": breakout_energy
        }

    def _analyze_volume(self, volume: List[float], indicators: Dict) -> Dict[str, Any]:
        avg_volume = sum(volume[-20:]) / 20
        current_volume = volume[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        obv = indicators["obv"]
        cmf = indicators["cmf"]
        mfi = indicators["mfi"]
        volume_score = 0.0
        signals = []
        if volume_ratio > self.config.volume_spike_threshold:
            volume_score += 15
            signals.append(f"Volume Spike ({volume_ratio:.1f}x)")
        if cmf > 0.1:
            volume_score += 10
            signals.append("Strong Money Flow In")
        elif cmf < -0.1:
            volume_score -= 10
            signals.append("Money Flow Out")
        if mfi < 20:
            volume_score += 15
            signals.append("MFI Oversold")
        elif mfi > 80:
            volume_score -= 15
            signals.append("MFI Overbought")
        return {
            "score": volume_score,
            "signals": signals,
            "volume_ratio": volume_ratio,
            "cmf": cmf,
            "mfi": mfi
        }

    def _calculate_signal_score(
        self,
        trend: Dict,
        momentum: Dict,
        volatility: Dict,
        volume: Dict,
        prebreakout: Dict,
        mode: StrategyMode
    ) -> tuple:
        base_score = (
            trend["strength"] * 25 +
            momentum["score"] +
            volatility["score"] +
            volume["score"]
        )
        prebreakout_score = prebreakout.get("prebreakout_score", 0)
        if prebreakout.get("stage") == "PRE_BREAKOUT":
            base_score += 20
        elif prebreakout.get("stage") == "BREAKOUT":
            base_score += 30
        if prebreakout_score > 70:
            base_score += (prebreakout_score - 70) * 0.5
        if mode == StrategyMode.SCALP:
            if volatility.get("squeeze_on"):
                base_score += 10
        elif mode == StrategyMode.PRE_BREAKOUT:
            base_score += prebreakout_score * 0.3
        reasons = []
        reasons.extend(momentum.get("signals", []))
        reasons.extend(volatility.get("signals", []))
        reasons.extend(volume.get("signals", []))
        reasons.append(f"Trend: {trend['direction']}")
        if prebreakout.get("stage") in ["PRE_BREAKOUT", "BREAKOUT"]:
            reasons.append(f"Stage: {prebreakout.get('stage')} ({prebreakout_score:.1f}%)")
        return base_score, reasons

    def _determine_signal_type(self, score: float, side: OrderSide) -> SignalType:
        if side == OrderSide.BUY:
            if score >= 85:
                return SignalType.STRONG_BUY
            elif score >= 70:
                return SignalType.BUY
            elif score >= 55:
                return SignalType.WEAK_BUY
            return SignalType.NEUTRAL
        else:
            if score >= 85:
                return SignalType.STRONG_SELL
            elif score >= 70:
                return SignalType.SELL
            elif score >= 55:
                return SignalType.WEAK_SELL
            return SignalType.NEUTRAL
