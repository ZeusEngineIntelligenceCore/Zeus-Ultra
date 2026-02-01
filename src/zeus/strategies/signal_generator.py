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
from ..indicators.advanced_math import AdvancedMathAnalyzer
from ..indicators.microstructure import MicrostructureAnalyzer, OrderBookSnapshot
from ..ml.regime_detector import RegimeDetector
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
    min_confidence: float = 70.0
    min_rr_ratio: float = 2.0
    max_spread_pct: float = 0.3
    atr_stop_mult: float = 1.8
    atr_tp_mult: float = 4.0
    scalp_atr_stop: float = 0.8
    scalp_atr_tp: float = 2.0
    swing_atr_stop: float = 2.0
    swing_atr_tp: float = 5.0
    rsi_oversold: float = 25.0
    rsi_overbought: float = 75.0
    rsi_extreme_oversold: float = 18.0
    rsi_extreme_overbought: float = 82.0
    volume_spike_threshold: float = 1.8
    volume_surge_threshold: float = 2.5
    trend_ema_fast: int = 8
    trend_ema_slow: int = 21
    trend_ema_trend: int = 55
    trend_ema_long: int = 200
    macd_histogram_threshold: float = 0.0002
    adx_strong_trend: float = 30.0
    adx_weak_trend: float = 20.0
    bb_squeeze_threshold: float = 0.02
    confluence_bonus: float = 15.0


class SignalGenerator:
    def __init__(self, config: Optional[SignalConfig] = None):
        self.config = config or SignalConfig()
        self.math = MathKernel()
        self.prebreakout = PreBreakoutDetector()
        self.mtf_analyzer = MultiTimeframeAnalyzer()
        self.advanced_math = AdvancedMathAnalyzer()
        self.regime_detector = RegimeDetector()
        self.microstructure = MicrostructureAnalyzer()

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
        advanced_analysis = self._analyze_advanced(high, low, close, volume)
        regime_analysis = self._analyze_regime(close)
        microstructure_analysis = self._analyze_microstructure(order_book_analysis) if order_book_analysis else None
        signal_score, signal_reasons = self._calculate_signal_score(
            trend, momentum, volatility, volume_analysis, prebreakout, mode,
            advanced_analysis, regime_analysis, microstructure_analysis
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
        ema_long = self.math.ema(close, min(self.config.trend_ema_long, len(close) - 1)) if len(close) > 50 else ema_trend
        current = close[-1]
        ema_spacing = abs(ema_fast - ema_slow) / current * 100 if current > 0 else 0
        ema_momentum = (ema_fast - ema_slow) / ema_slow * 100 if ema_slow > 0 else 0
        price_vs_ema = (current - ema_trend) / ema_trend * 100 if ema_trend > 0 else 0
        if ema_fast > ema_slow > ema_trend > ema_long and current > ema_fast:
            direction = "ULTRA_BULLISH"
            strength = 1.2 + min(ema_spacing * 0.05, 0.3)
        elif ema_fast > ema_slow > ema_trend and current > ema_fast:
            direction = "STRONG_UP"
            strength = 1.0 + min(ema_spacing * 0.04, 0.2)
        elif ema_fast > ema_slow and current > ema_slow:
            direction = "UP"
            strength = 0.7 + min(ema_momentum * 0.02, 0.15)
        elif ema_fast < ema_slow < ema_trend < ema_long and current < ema_fast:
            direction = "ULTRA_BEARISH"
            strength = -1.2 - min(ema_spacing * 0.05, 0.3)
        elif ema_fast < ema_slow < ema_trend and current < ema_fast:
            direction = "STRONG_DOWN"
            strength = -1.0 - min(ema_spacing * 0.04, 0.2)
        elif ema_fast < ema_slow and current < ema_slow:
            direction = "DOWN"
            strength = -0.7 - min(abs(ema_momentum) * 0.02, 0.15)
        elif abs(ema_fast - ema_slow) / current < 0.001:
            direction = "CONSOLIDATING"
            strength = 0.0
        else:
            direction = "NEUTRAL"
            strength = 0.0
        slope_5 = (close[-1] - close[-5]) / close[-5] * 100 if len(close) >= 5 and close[-5] > 0 else 0
        slope_10 = (close[-1] - close[-10]) / close[-10] * 100 if len(close) >= 10 and close[-10] > 0 else 0
        if slope_5 > 0 and slope_10 > 0 and slope_5 > slope_10:
            strength += 0.1
        elif slope_5 < 0 and slope_10 < 0 and slope_5 < slope_10:
            strength -= 0.1
        return {
            "direction": direction,
            "strength": strength,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "ema_trend": ema_trend,
            "ema_long": ema_long,
            "ema_spacing": ema_spacing,
            "ema_momentum": ema_momentum,
            "price_vs_ema": price_vs_ema,
            "slope_5": slope_5,
            "slope_10": slope_10
        }

    def _analyze_momentum(self, indicators: Dict) -> Dict[str, Any]:
        rsi = indicators["rsi"]
        macd_line, signal_line, histogram = indicators["macd"]
        stoch_k, stoch_d = indicators["stoch_rsi"]
        adx, plus_di, minus_di = indicators["adx"]
        momentum_score = 0.0
        signals = []
        confluence_count = 0
        if rsi < self.config.rsi_extreme_oversold:
            momentum_score += 40
            signals.append("RSI Extreme Oversold - High Probability Reversal")
            confluence_count += 2
        elif rsi < self.config.rsi_oversold:
            momentum_score += 30
            signals.append("RSI Oversold")
            confluence_count += 1
        elif rsi > self.config.rsi_extreme_overbought:
            momentum_score -= 40
            signals.append("RSI Extreme Overbought - High Probability Reversal")
            confluence_count += 2
        elif rsi > self.config.rsi_overbought:
            momentum_score -= 30
            signals.append("RSI Overbought")
            confluence_count += 1
        elif 45 < rsi < 55:
            momentum_score += 5
        macd_strength = abs(histogram) / abs(macd_line) * 100 if macd_line != 0 else 0
        if histogram > 0 and macd_line > signal_line:
            base_score = 20
            if macd_strength > 20:
                base_score += 10
                signals.append("MACD Strong Bullish Momentum")
            else:
                signals.append("MACD Bullish")
            momentum_score += base_score
            confluence_count += 1
        elif histogram < 0 and macd_line < signal_line:
            base_score = -20
            if macd_strength > 20:
                base_score -= 10
                signals.append("MACD Strong Bearish Momentum")
            else:
                signals.append("MACD Bearish")
            momentum_score += base_score
            confluence_count += 1
        di_diff = plus_di - minus_di
        if adx > self.config.adx_strong_trend:
            trend_mult = 1.0 + (adx - 30) / 50
            if di_diff > 10:
                momentum_score += int(20 * trend_mult)
                signals.append(f"Strong Bullish Trend (ADX:{adx:.0f})")
                confluence_count += 1
            elif di_diff < -10:
                momentum_score -= int(20 * trend_mult)
                signals.append(f"Strong Bearish Trend (ADX:{adx:.0f})")
                confluence_count += 1
        elif adx > self.config.adx_weak_trend:
            if di_diff > 5:
                momentum_score += 10
                signals.append("Moderate Bullish Trend")
            elif di_diff < -5:
                momentum_score -= 10
                signals.append("Moderate Bearish Trend")
        stoch_cross_up = stoch_k > stoch_d and stoch_k < 30
        stoch_cross_down = stoch_k < stoch_d and stoch_k > 70
        if stoch_k < 15:
            momentum_score += 20
            signals.append("Stochastic Extreme Oversold")
            confluence_count += 1
        elif stoch_k < 20:
            momentum_score += 12
            signals.append("Stochastic Oversold")
        elif stoch_k > 85:
            momentum_score -= 20
            signals.append("Stochastic Extreme Overbought")
            confluence_count += 1
        elif stoch_k > 80:
            momentum_score -= 12
            signals.append("Stochastic Overbought")
        if stoch_cross_up:
            momentum_score += 8
            signals.append("Stochastic Bullish Cross")
        elif stoch_cross_down:
            momentum_score -= 8
            signals.append("Stochastic Bearish Cross")
        if confluence_count >= 3:
            momentum_score += self.config.confluence_bonus
            signals.append(f"High Confluence ({confluence_count} signals)")
        return {
            "score": momentum_score,
            "signals": signals,
            "rsi": rsi,
            "macd_histogram": histogram,
            "macd_strength": macd_strength,
            "adx": adx,
            "di_diff": di_diff,
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
            "confluence_count": confluence_count
        }

    def _analyze_volatility(self, indicators: Dict, high: List[float], 
                           low: List[float], close: List[float]) -> Dict[str, Any]:
        squeeze_on, momentum, bb_width = indicators["ttm_squeeze"]
        atr = indicators["atr"]
        bb_upper, bb_mid, bb_lower = indicators["bollinger"]
        current = close[-1]
        volatility_score = 0.0
        signals = []
        bb_position = (current - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
        is_tight_squeeze = bb_width < self.config.bb_squeeze_threshold
        if squeeze_on and momentum > 0:
            base_score = 30
            if is_tight_squeeze:
                base_score += 10
                signals.append("TTM Squeeze PRIMED - Tight BB + Bullish Momentum")
            else:
                signals.append("TTM Squeeze Bullish Fire")
            volatility_score += base_score
        elif squeeze_on and momentum < 0:
            volatility_score -= 18
            signals.append("TTM Squeeze Bearish")
        elif not squeeze_on and momentum > 0:
            volatility_score += 15
            signals.append("Squeeze Released Bullish - Expansion Phase")
        elif not squeeze_on and momentum < 0:
            volatility_score -= 12
            signals.append("Squeeze Released Bearish")
        bb_touch_lower = current <= bb_lower * 1.002
        bb_touch_upper = current >= bb_upper * 0.998
        if bb_touch_lower:
            recent_touches = sum(1 for i in range(-5, 0) if close[i] <= bb_lower * 1.005)
            if recent_touches >= 2:
                volatility_score += 25
                signals.append("Multiple BB Lower Band Touches - Strong Support")
            else:
                volatility_score += 18
                signals.append("Price at Lower BB - Potential Bounce")
        elif bb_touch_upper:
            recent_touches = sum(1 for i in range(-5, 0) if close[i] >= bb_upper * 0.995)
            if recent_touches >= 2:
                volatility_score -= 25
                signals.append("Multiple BB Upper Band Touches - Strong Resistance")
            else:
                volatility_score -= 18
                signals.append("Price at Upper BB - Potential Pullback")
        elif bb_position < 0.2:
            volatility_score += 12
            signals.append("Price Near Lower BB")
        elif bb_position > 0.8:
            volatility_score -= 12
            signals.append("Price Near Upper BB")
        breakout_energy = indicators["breakout_energy"]
        if breakout_energy > 80:
            volatility_score += 30
            signals.append(f"EXTREME Breakout Energy ({breakout_energy:.1f}) - Imminent Move")
        elif breakout_energy > 60:
            volatility_score += 20
            signals.append(f"High Breakout Energy ({breakout_energy:.1f})")
        elif breakout_energy > 40:
            volatility_score += 10
            signals.append(f"Moderate Breakout Energy ({breakout_energy:.1f})")
        atr_pct = atr / current * 100 if current > 0 else 0
        return {
            "score": volatility_score,
            "signals": signals,
            "squeeze_on": squeeze_on,
            "is_tight_squeeze": is_tight_squeeze,
            "atr": atr,
            "atr_pct": atr_pct,
            "bb_position": bb_position,
            "bb_width": bb_width,
            "breakout_energy": breakout_energy
        }

    def _analyze_volume(self, volume: List[float], indicators: Dict) -> Dict[str, Any]:
        avg_volume_20 = sum(volume[-20:]) / 20 if len(volume) >= 20 else sum(volume) / len(volume)
        avg_volume_5 = sum(volume[-5:]) / 5 if len(volume) >= 5 else avg_volume_20
        current_volume = volume[-1]
        volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1.0
        volume_trend = avg_volume_5 / avg_volume_20 if avg_volume_20 > 0 else 1.0
        obv = indicators["obv"]
        cmf = indicators["cmf"]
        mfi = indicators["mfi"]
        volume_score = 0.0
        signals = []
        if volume_ratio > self.config.volume_surge_threshold:
            volume_score += 25
            signals.append(f"VOLUME SURGE ({volume_ratio:.1f}x) - Institutional Interest")
        elif volume_ratio > self.config.volume_spike_threshold:
            volume_score += 15
            signals.append(f"Volume Spike ({volume_ratio:.1f}x)")
        elif volume_ratio > 1.3:
            volume_score += 8
            signals.append(f"Above Average Volume ({volume_ratio:.1f}x)")
        if volume_trend > 1.5:
            volume_score += 10
            signals.append("Rising Volume Trend")
        elif volume_trend < 0.7:
            volume_score -= 5
            signals.append("Declining Volume Trend")
        if cmf > 0.2:
            volume_score += 18
            signals.append("Very Strong Money Flow In")
        elif cmf > 0.1:
            volume_score += 12
            signals.append("Strong Money Flow In")
        elif cmf > 0.05:
            volume_score += 6
            signals.append("Positive Money Flow")
        elif cmf < -0.2:
            volume_score -= 18
            signals.append("Very Strong Money Flow Out")
        elif cmf < -0.1:
            volume_score -= 12
            signals.append("Strong Money Flow Out")
        elif cmf < -0.05:
            volume_score -= 6
            signals.append("Negative Money Flow")
        if mfi < 15:
            volume_score += 22
            signals.append("MFI Extreme Oversold - High Reversal Probability")
        elif mfi < 20:
            volume_score += 15
            signals.append("MFI Oversold")
        elif mfi > 85:
            volume_score -= 22
            signals.append("MFI Extreme Overbought - High Reversal Probability")
        elif mfi > 80:
            volume_score -= 15
            signals.append("MFI Overbought")
        obv_trend = 0
        if len(volume) >= 10:
            obv_ma = sum(volume[-10:]) / 10
            obv_current = sum(volume[-3:]) / 3
            if obv_current > obv_ma * 1.1:
                obv_trend = 1
                volume_score += 8
                signals.append("OBV Rising")
            elif obv_current < obv_ma * 0.9:
                obv_trend = -1
                volume_score -= 8
                signals.append("OBV Falling")
        return {
            "score": volume_score,
            "signals": signals,
            "volume_ratio": volume_ratio,
            "volume_trend": volume_trend,
            "cmf": cmf,
            "mfi": mfi,
            "obv": obv,
            "obv_trend": obv_trend
        }

    def _analyze_advanced(self, high: List[float], low: List[float], 
                         close: List[float], volume: List[float]) -> Dict[str, Any]:
        try:
            analysis = self.advanced_math.analyze(high, low, close, volume)
            return {
                "hurst": analysis.hurst_exponent,
                "trend_persistence": analysis.trend_persistence,
                "nearest_support": analysis.nearest_support,
                "nearest_resistance": analysis.nearest_resistance,
                "entropy": analysis.market_entropy,
                "efficiency": analysis.market_efficiency,
                "denoised_trend": analysis.denoised_trend,
                "score": analysis.overall_score,
                "signals": analysis.signals
            }
        except Exception:
            return {"score": 50, "signals": [], "hurst": 0.5}
    
    def _analyze_regime(self, close: List[float]) -> Dict[str, Any]:
        try:
            for price in close[-50:]:
                self.regime_detector.update(price)
            state = self.regime_detector.detect_regime()
            return {
                "regime": state.regime.value,
                "trading_mode": state.trading_mode.value,
                "confidence": state.confidence,
                "volatility_state": state.volatility_state,
                "trend_strength": state.trend_strength,
                "position_size_mult": state.recommended_position_size_mult,
                "stop_mult": state.recommended_stop_mult,
                "signals": state.signals
            }
        except Exception:
            return {"regime": "ranging", "trading_mode": "defensive", "confidence": 0.5, "signals": []}
    
    def _analyze_microstructure(self, order_book: Dict) -> Dict:
        if not order_book:
            return {}
        bid_volume = order_book.get("total_bid_volume", 0)
        ask_volume = order_book.get("total_ask_volume", 0)
        spread_pct = order_book.get("spread_pct", 0)
        order_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0
        liquidity_score = min(100, (bid_volume + ask_volume) / 1000 * 10)
        signals = []
        if order_imbalance > 0.3:
            signals.append("Strong buy pressure (OFI)")
        elif order_imbalance < -0.3:
            signals.append("Strong sell pressure (OFI)")
        if spread_pct < 0.1:
            signals.append("Tight spread (high liquidity)")
        elif spread_pct > 0.5:
            signals.append("Wide spread (low liquidity)")
        return {
            "order_imbalance": order_imbalance,
            "liquidity_score": liquidity_score,
            "spread_pct": spread_pct,
            "bid_volume": bid_volume,
            "ask_volume": ask_volume,
            "signals": signals
        }

    def _calculate_signal_score(
        self,
        trend: Dict,
        momentum: Dict,
        volatility: Dict,
        volume: Dict,
        prebreakout: Dict,
        mode: StrategyMode,
        advanced: Optional[Dict] = None,
        regime: Optional[Dict] = None,
        microstructure: Optional[Dict] = None
    ) -> tuple:
        trend_weight = 30
        momentum_weight = 1.2
        volatility_weight = 1.1
        volume_weight = 1.0
        base_score = (
            trend["strength"] * trend_weight +
            momentum["score"] * momentum_weight +
            volatility["score"] * volatility_weight +
            volume["score"] * volume_weight
        )
        prebreakout_score = prebreakout.get("prebreakout_score", 0)
        if prebreakout.get("stage") == "PRE_BREAKOUT":
            base_score += 25
        elif prebreakout.get("stage") == "BREAKOUT":
            base_score += 40
        elif prebreakout.get("stage") == "ACCUMULATION":
            base_score += 10
        if prebreakout_score > 80:
            base_score += (prebreakout_score - 80) * 0.8
        elif prebreakout_score > 70:
            base_score += (prebreakout_score - 70) * 0.5
        trend_dir = trend.get("direction", "NEUTRAL")
        mom_score = momentum.get("score", 0)
        if trend_dir in ["STRONG_UP", "ULTRA_BULLISH"] and mom_score > 30:
            base_score += 15
        elif trend_dir in ["STRONG_DOWN", "ULTRA_BEARISH"] and mom_score < -30:
            base_score -= 15
        if volatility.get("is_tight_squeeze") and prebreakout_score > 60:
            base_score += 20
        if volume.get("volume_ratio", 1.0) > 2.0 and prebreakout_score > 50:
            base_score += 15
        confluence = momentum.get("confluence_count", 0)
        if confluence >= 4:
            base_score += 25
        elif confluence >= 3:
            base_score += 15
        if mode == StrategyMode.SCALP:
            if volatility.get("squeeze_on"):
                base_score += 15
            if volatility.get("atr_pct", 0) > 2:
                base_score += 10
        elif mode == StrategyMode.PRE_BREAKOUT:
            base_score += prebreakout_score * 0.4
        elif mode == StrategyMode.SWING:
            if trend_dir in ["STRONG_UP", "ULTRA_BULLISH", "STRONG_DOWN", "ULTRA_BEARISH"]:
                base_score += 10
        if advanced:
            adv_score = advanced.get("score", 50)
            base_score += (adv_score - 50) * 0.3
            hurst = advanced.get("hurst", 0.5)
            if hurst > 0.65:
                base_score += 10
            elif hurst < 0.35:
                base_score -= 5
            if advanced.get("denoised_trend") == "bullish":
                base_score += 5
            elif advanced.get("denoised_trend") == "bearish":
                base_score -= 5
        if regime:
            regime_confidence = regime.get("confidence", 0.5)
            trading_mode = regime.get("trading_mode", "defensive")
            if trading_mode == "momentum" and trend_dir in ["STRONG_UP", "ULTRA_BULLISH"]:
                base_score += 15 * regime_confidence
            elif trading_mode == "mean_reversion" and momentum.get("score", 0) < -20:
                base_score += 10 * regime_confidence
            elif trading_mode == "breakout" and prebreakout_score > 70:
                base_score += 20 * regime_confidence
            elif trading_mode == "defensive":
                base_score *= 0.85
        if microstructure:
            ofi = microstructure.get("order_imbalance", 0)
            liquidity = microstructure.get("liquidity_score", 50)
            spread = microstructure.get("spread_pct", 0.5)
            if ofi > 0.3 and trend_dir in ["STRONG_UP", "ULTRA_BULLISH"]:
                base_score += 15
            elif ofi < -0.3 and trend_dir in ["STRONG_DOWN", "ULTRA_BEARISH"]:
                base_score -= 15
            if liquidity > 70:
                base_score += 5
            elif liquidity < 30:
                base_score -= 10
            if spread > 0.5:
                base_score -= 5
        reasons = []
        reasons.extend(momentum.get("signals", []))
        reasons.extend(volatility.get("signals", []))
        reasons.extend(volume.get("signals", []))
        reasons.append(f"Trend: {trend['direction']} (Str: {trend['strength']:.2f})")
        if prebreakout.get("stage") in ["PRE_BREAKOUT", "BREAKOUT", "ACCUMULATION"]:
            reasons.append(f"Stage: {prebreakout.get('stage')} ({prebreakout_score:.1f}%)")
        if confluence >= 3:
            reasons.append(f"Multi-Indicator Confluence: {confluence}")
        if advanced:
            reasons.extend(advanced.get("signals", [])[:2])
        if regime:
            reasons.extend(regime.get("signals", [])[:2])
        if microstructure:
            reasons.extend(microstructure.get("signals", [])[:2])
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
