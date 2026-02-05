#!/usr/bin/env python3
"""
ZEUS BREAKOUT & FAKEOUT ANALYZER
Enhanced breakout quality scoring with volume profile, momentum confirmation,
and sophisticated fakeout detection
"""

from __future__ import annotations
import math
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger("Zeus.BreakoutAnalyzer")


class BreakoutQuality(Enum):
    HIGH_QUALITY = "high_quality"
    MODERATE = "moderate"
    WEAK = "weak"
    LIKELY_FAKEOUT = "likely_fakeout"
    CONFIRMED_FAKEOUT = "confirmed_fakeout"


@dataclass
class BreakoutSignal:
    quality: BreakoutQuality
    score: float  # 0-100
    is_fakeout: bool
    fakeout_probability: float  # 0-1
    confirmation_count: int
    volume_quality: str
    momentum_quality: str
    resistance_break: bool
    support_break: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VolumeSpike:
    detected: bool
    strength: float  # 0-10 scale
    type: str  # "bullish", "bearish", "neutral"
    sustainability: float  # 0-1 how likely to sustain
    anomaly_score: float


@dataclass
class VolumeFallout:
    detected: bool
    severity: float  # 0-1
    duration_bars: int
    warning_level: str  # "critical", "warning", "minor"


class BreakoutAnalyzer:
    def __init__(self):
        self.recent_breakouts: Dict[str, List[Dict]] = {}  # symbol -> breakout history
    
    def analyze_breakout_quality(
        self,
        high: List[float],
        low: List[float],
        close: List[float],
        volume: List[float],
        lookback: int = 20
    ) -> BreakoutSignal:
        if len(close) < lookback + 10:
            return BreakoutSignal(
                quality=BreakoutQuality.WEAK,
                score=0,
                is_fakeout=False,
                fakeout_probability=0.5,
                confirmation_count=0,
                volume_quality="unknown",
                momentum_quality="unknown",
                resistance_break=False,
                support_break=False
            )
        
        current = close[-1]
        prev_high = max(high[-lookback-1:-1])
        prev_low = min(low[-lookback-1:-1])
        avg_range = sum(high[i] - low[i] for i in range(-lookback, 0)) / lookback
        
        resistance_break = current > prev_high
        support_break = current < prev_low
        
        if not resistance_break and not support_break:
            return BreakoutSignal(
                quality=BreakoutQuality.WEAK,
                score=30,
                is_fakeout=False,
                fakeout_probability=0.3,
                confirmation_count=0,
                volume_quality="normal",
                momentum_quality="neutral",
                resistance_break=False,
                support_break=False
            )
        
        volume_score = self._analyze_volume_profile(volume, lookback)
        momentum_score = self._analyze_momentum_confirmation(close, high, low, lookback)
        fakeout_prob = self._calculate_fakeout_probability(
            close, high, low, volume, prev_high, prev_low, lookback
        )
        
        confirmations = self._count_confirmations(
            close, high, low, volume, prev_high, prev_low, lookback
        )
        
        breakout_score = 50.0
        breakout_score += volume_score * 20
        breakout_score += momentum_score * 20
        breakout_score -= fakeout_prob * 30
        breakout_score += confirmations * 5
        
        breakout_score = max(0, min(100, breakout_score))
        
        if breakout_score >= 75 and fakeout_prob < 0.3:
            quality = BreakoutQuality.HIGH_QUALITY
        elif breakout_score >= 55 and fakeout_prob < 0.5:
            quality = BreakoutQuality.MODERATE
        elif fakeout_prob >= 0.7:
            quality = BreakoutQuality.CONFIRMED_FAKEOUT
        elif fakeout_prob >= 0.5:
            quality = BreakoutQuality.LIKELY_FAKEOUT
        else:
            quality = BreakoutQuality.WEAK
        
        volume_quality = "strong" if volume_score > 0.7 else "moderate" if volume_score > 0.4 else "weak"
        momentum_quality = "strong" if momentum_score > 0.7 else "moderate" if momentum_score > 0.4 else "weak"
        
        return BreakoutSignal(
            quality=quality,
            score=breakout_score,
            is_fakeout=fakeout_prob >= 0.6,
            fakeout_probability=fakeout_prob,
            confirmation_count=confirmations,
            volume_quality=volume_quality,
            momentum_quality=momentum_quality,
            resistance_break=resistance_break,
            support_break=support_break,
            details={
                "volume_score": round(volume_score, 2),
                "momentum_score": round(momentum_score, 2),
                "prev_high": round(prev_high, 8),
                "prev_low": round(prev_low, 8),
                "breakout_distance": round((current - prev_high) / prev_high * 100 if resistance_break else (prev_low - current) / prev_low * 100, 2)
            }
        )
    
    def _analyze_volume_profile(self, volume: List[float], lookback: int) -> float:
        if len(volume) < lookback + 5:
            return 0.5
        
        avg_volume = sum(volume[-lookback-1:-1]) / lookback
        current_volume = volume[-1]
        recent_avg = sum(volume[-3:]) / 3
        
        volume_ratio = current_volume / (avg_volume + 1e-10)
        recent_ratio = recent_avg / (avg_volume + 1e-10)
        
        score = 0.0
        if volume_ratio > 2.0:
            score += 0.4
        elif volume_ratio > 1.5:
            score += 0.25
        elif volume_ratio > 1.2:
            score += 0.15
        
        if recent_ratio > 1.5:
            score += 0.3
        elif recent_ratio > 1.2:
            score += 0.2
        
        vol_trend = []
        for i in range(-5, 0):
            vol_trend.append(volume[i])
        increasing = all(vol_trend[i] <= vol_trend[i+1] for i in range(len(vol_trend)-1))
        if increasing:
            score += 0.3
        
        return min(1.0, score)
    
    def _analyze_momentum_confirmation(
        self,
        close: List[float],
        high: List[float],
        low: List[float],
        lookback: int
    ) -> float:
        if len(close) < lookback + 5:
            return 0.5
        
        returns = [(close[i] - close[i-1]) / close[i-1] for i in range(-5, 0)]
        positive_returns = sum(1 for r in returns if r > 0)
        
        gains = [max(0, close[i] - close[i-1]) for i in range(-14, 0)]
        losses = [max(0, close[i-1] - close[i]) for i in range(-14, 0)]
        avg_gain = sum(gains) / 14
        avg_loss = sum(losses) / 14
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        body = close[-1] - close[-2]
        range_size = high[-1] - low[-1]
        body_to_range = abs(body) / (range_size + 1e-10)
        
        score = 0.0
        
        if positive_returns >= 4:
            score += 0.35
        elif positive_returns >= 3:
            score += 0.2
        
        if 60 <= rsi <= 80:
            score += 0.3
        elif 55 <= rsi <= 85:
            score += 0.15
        
        if body_to_range > 0.7 and body > 0:
            score += 0.35
        elif body_to_range > 0.5 and body > 0:
            score += 0.2
        
        return min(1.0, score)
    
    def _calculate_fakeout_probability(
        self,
        close: List[float],
        high: List[float],
        low: List[float],
        volume: List[float],
        prev_high: float,
        prev_low: float,
        lookback: int
    ) -> float:
        fakeout_score = 0.0
        
        current = close[-1]
        if current > prev_high:
            penetration = (current - prev_high) / (high[-1] - low[-1] + 1e-10)
            if penetration < 0.3:
                fakeout_score += 0.3
        
        avg_volume = sum(volume[-lookback-1:-1]) / lookback
        if volume[-1] < avg_volume * 0.8:
            fakeout_score += 0.25
        
        body = abs(close[-1] - close[-2])
        upper_wick = high[-1] - max(close[-1], close[-2])
        lower_wick = min(close[-1], close[-2]) - low[-1]
        total_range = high[-1] - low[-1]
        
        if upper_wick > body and current > prev_high:
            fakeout_score += 0.2
        
        recent_fakeouts = 0
        for i in range(-10, -1):
            if high[i] > prev_high and close[i] < prev_high:
                recent_fakeouts += 1
        fakeout_score += recent_fakeouts * 0.1
        
        if close[-2] > prev_high and close[-1] < prev_high:
            fakeout_score += 0.3
        
        return min(1.0, fakeout_score)
    
    def _count_confirmations(
        self,
        close: List[float],
        high: List[float],
        low: List[float],
        volume: List[float],
        prev_high: float,
        prev_low: float,
        lookback: int
    ) -> int:
        confirmations = 0
        
        bars_above = sum(1 for i in range(-3, 0) if close[i] > prev_high)
        if bars_above >= 2:
            confirmations += 1
        
        avg_volume = sum(volume[-lookback:]) / lookback
        if sum(volume[-3:]) / 3 > avg_volume * 1.3:
            confirmations += 1
        
        if all(close[i] > close[i-1] for i in range(-2, 0)):
            confirmations += 1
        
        if close[-1] > prev_high * 1.005:
            confirmations += 1
        
        higher_lows = all(low[i] > low[i-1] for i in range(-3, 0))
        if higher_lows:
            confirmations += 1
        
        return confirmations
    
    def detect_volume_spike(
        self,
        volume: List[float],
        close: List[float],
        lookback: int = 20
    ) -> VolumeSpike:
        if len(volume) < lookback + 5:
            return VolumeSpike(False, 0, "neutral", 0, 0)
        
        avg_volume = sum(volume[-lookback-1:-1]) / lookback
        std_volume = math.sqrt(sum((v - avg_volume) ** 2 for v in volume[-lookback-1:-1]) / lookback)
        
        current_volume = volume[-1]
        z_score = (current_volume - avg_volume) / (std_volume + 1e-10)
        
        if z_score < 2.0:
            return VolumeSpike(False, 0, "neutral", 0, z_score)
        
        strength = min(10.0, z_score)
        
        price_change = (close[-1] - close[-2]) / close[-2]
        if price_change > 0.005:
            spike_type = "bullish"
        elif price_change < -0.005:
            spike_type = "bearish"
        else:
            spike_type = "neutral"
        
        follow_through = 0
        for i in range(-3, -1):
            if volume[i] > avg_volume * 1.5:
                follow_through += 1
        
        sustainability = min(1.0, follow_through / 2 + (z_score - 2) / 5)
        
        return VolumeSpike(
            detected=True,
            strength=strength,
            type=spike_type,
            sustainability=sustainability,
            anomaly_score=z_score
        )
    
    def detect_volume_fallout(
        self,
        volume: List[float],
        lookback: int = 20
    ) -> VolumeFallout:
        if len(volume) < lookback + 5:
            return VolumeFallout(False, 0, 0, "minor")
        
        avg_volume = sum(volume[-lookback-1:-1]) / lookback
        
        low_volume_bars = 0
        for i in range(-5, 0):
            if volume[i] < avg_volume * 0.5:
                low_volume_bars += 1
        
        if low_volume_bars < 3:
            return VolumeFallout(False, 0, 0, "minor")
        
        current_ratio = volume[-1] / (avg_volume + 1e-10)
        recent_ratio = sum(volume[-3:]) / 3 / (avg_volume + 1e-10)
        
        severity = 1.0 - min(1.0, recent_ratio)
        
        if severity > 0.7 or low_volume_bars >= 5:
            warning_level = "critical"
        elif severity > 0.4 or low_volume_bars >= 4:
            warning_level = "warning"
        else:
            warning_level = "minor"
        
        return VolumeFallout(
            detected=True,
            severity=severity,
            duration_bars=low_volume_bars,
            warning_level=warning_level
        )
    
    def is_fakeout_reversal(
        self,
        close: List[float],
        high: List[float],
        low: List[float],
        volume: List[float],
        entry_price: float,
        breakout_direction: str  # "up" or "down"
    ) -> Tuple[bool, float, str]:
        if len(close) < 10:
            return False, 0.0, "insufficient_data"
        
        fakeout_signals = 0
        reasons = []
        
        if breakout_direction == "up":
            if close[-1] < entry_price:
                fakeout_signals += 2
                reasons.append("price_below_entry")
            
            peak = max(high[-5:])
            if peak > entry_price and close[-1] < entry_price * 0.995:
                fakeout_signals += 2
                reasons.append("false_breakout_reversal")
            
            if close[-1] < close[-2] < close[-3]:
                fakeout_signals += 1
                reasons.append("declining_closes")
        else:
            if close[-1] > entry_price:
                fakeout_signals += 2
                reasons.append("price_above_entry")
            
            trough = min(low[-5:])
            if trough < entry_price and close[-1] > entry_price * 1.005:
                fakeout_signals += 2
                reasons.append("false_breakdown_reversal")
        
        avg_volume = sum(volume[-20:-5]) / 15 if len(volume) >= 20 else sum(volume[:-5]) / max(1, len(volume) - 5)
        if sum(volume[-3:]) / 3 < avg_volume * 0.6:
            fakeout_signals += 1
            reasons.append("volume_collapse")
        
        fakeout_probability = min(1.0, fakeout_signals / 5)
        is_fakeout = fakeout_probability >= 0.5
        
        return is_fakeout, fakeout_probability, ", ".join(reasons) if reasons else "none"
    
    def get_whale_volume_signals(
        self,
        volume: List[float],
        close: List[float],
        trade_sizes: Optional[List[float]] = None,
        lookback: int = 50
    ) -> Dict[str, Any]:
        if len(volume) < lookback:
            return {"whale_activity": "unknown", "signals": []}
        
        signals = []
        
        avg_vol = sum(volume[-lookback:]) / lookback
        std_vol = math.sqrt(sum((v - avg_vol) ** 2 for v in volume[-lookback:]) / lookback)
        
        extreme_volumes = []
        for i in range(-lookback, 0):
            z = (volume[i] - avg_vol) / (std_vol + 1e-10)
            if z > 3.0:
                extreme_volumes.append({
                    "bar": i,
                    "z_score": round(z, 2),
                    "volume": volume[i],
                    "price_change": round((close[i] - close[i-1]) / close[i-1] * 100, 2) if i > -lookback else 0
                })
        
        if extreme_volumes:
            signals.append({
                "type": "whale_accumulation",
                "count": len(extreme_volumes),
                "details": extreme_volumes[-5:]  # Last 5
            })
        
        up_volume = sum(volume[i] for i in range(-20, 0) if close[i] > close[i-1])
        down_volume = sum(volume[i] for i in range(-20, 0) if close[i] < close[i-1])
        imbalance = (up_volume - down_volume) / (up_volume + down_volume + 1e-10)
        
        if abs(imbalance) > 0.3:
            signals.append({
                "type": "volume_imbalance",
                "direction": "bullish" if imbalance > 0 else "bearish",
                "strength": round(abs(imbalance), 2)
            })
        
        activity_level = "high" if len(extreme_volumes) >= 3 else "moderate" if extreme_volumes else "low"
        
        return {
            "whale_activity": activity_level,
            "signals": signals,
            "avg_volume": round(avg_vol, 2),
            "volume_std": round(std_vol, 2),
            "extreme_volume_count": len(extreme_volumes),
            "volume_imbalance": round(imbalance, 3)
        }
