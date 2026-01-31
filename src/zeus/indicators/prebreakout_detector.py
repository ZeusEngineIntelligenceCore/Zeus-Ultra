#!/usr/bin/env python3
"""
ZEUS PRE-BREAKOUT DETECTOR - CAT-9 Ultra Pattern Recognition
Advanced scoring system for detecting pre-breakout conditions
"""

from __future__ import annotations
import math
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .math_kernel import MathKernel


def clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def tanh01(x: float) -> float:
    return (math.tanh(x) + 1) / 2


def sigmoid(x: float) -> float:
    if x > 60:
        return 1.0
    if x < -60:
        return 0.0
    return 1 / (1 + math.exp(-x))


def safe_mean(vals: List[float], default: float = 0.0) -> float:
    valid = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
    return sum(valid) / len(valid) if valid else default


def safe_std(vals: List[float], default: float = 0.0) -> float:
    if len(vals) < 2:
        return default
    m = safe_mean(vals)
    variance = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return math.sqrt(variance)


def pct_change(seq: List[float]) -> List[float]:
    out = []
    for i in range(1, len(seq)):
        prev = seq[i - 1] if seq[i - 1] != 0 else 1.0
        out.append((seq[i] - seq[i - 1]) / abs(prev))
    return out


def linear_slope(seq: List[float]) -> float:
    n = len(seq)
    if n < 2:
        return 0.0
    x = list(range(n))
    mx = (n - 1) / 2
    my = safe_mean(seq)
    num = sum((x[i] - mx) * (seq[i] - my) for i in range(n))
    den = sum((xi - mx) ** 2 for xi in x) or 1.0
    return num / den


@dataclass
class BreakoutConfig:
    rsi_period: int = 14
    spike_cap: float = 0.25
    pressure_cap: float = 1.8
    impulse_cap: float = 6.0
    liquidity_scale: float = 0.6
    break_pre: float = 70.0
    break_out: float = 85.0
    ladder_tiers: int = 3
    ladder_step_atr: float = 0.6
    ladder_sell_step: float = 0.8
    weights: Optional[Dict[str, float]] = None

    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                "rsi": 0.12,
                "momentum_cf": 0.10,
                "vol_spike": 0.10,
                "pressure": 0.12,
                "microtrend": 0.08,
                "accel": 0.06,
                "anomaly_vol": 0.05,
                "candle_proj": 0.07,
                "consistency": 0.08,
                "impulse": 0.10,
                "liquidity": 0.06,
                "squeeze": 0.06
            }


class PreBreakoutDetector:
    def __init__(self, config: Optional[BreakoutConfig] = None):
        self.cfg = config or BreakoutConfig()
        self.math = MathKernel()

    async def predictive_rsi(self, prices: List[float]) -> float:
        if len(prices) < 4:
            return 0.5
        rsi = self.math.rsi(prices, self.cfg.rsi_period)
        normalized = (rsi - 30) / 40
        return clip01(normalized)

    async def momentum_cf(self, prices: List[float]) -> float:
        if len(prices) < 4:
            return 0.5
        slope = linear_slope(prices[-12:])
        return clip01(tanh01(slope * 6))

    async def vol_spike(self, prices: List[float]) -> float:
        if len(prices) < 4:
            return 0.0
        lo, hi = min(prices), max(prices)
        med = sorted(prices)[len(prices) // 2]
        raw = (hi - lo) / abs(med or 1e-9)
        return clip01(raw / self.cfg.spike_cap)

    async def pressure(self, vols: List[float], prices: List[float]) -> float:
        if len(prices) < 4:
            return 0.0
        rets = pct_change(prices)
        last = rets[-1] if rets else 0.0
        med = sorted(vols)[len(vols) // 2] if vols else 1e-9
        vol_ratio = safe_mean(vols[-3:]) / (med or 1e-9)
        raw = abs(last) * vol_ratio
        return clip01(raw / self.cfg.pressure_cap)

    async def microtrend(self, prices: List[float]) -> float:
        return clip01(tanh01(linear_slope(prices[-15:]) * 8))

    async def accel(self, prices: List[float]) -> float:
        rets = pct_change(prices)
        if len(rets) < 4:
            return 0.5
        dif = [rets[i] - rets[i - 1] for i in range(1, len(rets))]
        if len(dif) < 2:
            return 0.5
        jerk = [dif[i] - dif[i - 1] for i in range(1, len(dif))]
        return clip01(tanh01(safe_mean(jerk) * 20))

    async def anomaly_vol(self, vols: List[float]) -> float:
        if len(vols) < 4:
            return 0.0
        med = sorted(vols)[len(vols) // 2]
        sd = safe_std(vols) or 1e-9
        z = (vols[-1] - med) / sd
        return clip01(sigmoid(z / 2))

    async def candle_proj(self, prices: List[float]) -> float:
        if len(prices) < 4:
            return 0.5
        bodies = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
        slope = linear_slope(bodies[-8:])
        wick = (max(prices) - min(prices)) - safe_mean(bodies)
        denom = safe_mean(bodies or [1e-9]) + 1.0
        return clip01(tanh01(slope * 5) * tanh01(wick / denom))

    async def consistency(self, prices: List[float]) -> float:
        if len(prices) < 8:
            return 0.5
        seg = max(4, len(prices) // 4)
        slopes = []
        for i in range(seg, len(prices) + 1, seg):
            slopes.append(linear_slope(prices[max(0, i - seg):i]))
        m = safe_mean(slopes)
        sd = safe_std(slopes)
        return clip01(1.0 - (sd / (abs(m) + 1e-9)))

    async def impulse(self, prices: List[float]) -> float:
        rets = pct_change(prices)
        if len(rets) < 4:
            return 0.0
        m = safe_mean(rets)
        sd = safe_std(rets) or 1e-9
        z = abs((rets[-1] - m) / sd)
        return clip01(sigmoid(clamp(z, 0.0, self.cfg.impulse_cap) - 1.0))

    async def liquidity_shift(self, liq: List[float]) -> float:
        if len(liq) < 4:
            return 0.5
        med = sorted(liq)[len(liq) // 2]
        ratio = (liq[-1] / (med or 1e-9)) - 1.0
        return clip01(tanh01(ratio / self.cfg.liquidity_scale))

    async def squeeze_pressure(self, high: List[float], low: List[float], close: List[float]) -> float:
        squeeze_on, momentum, bb_width = self.math.ttm_squeeze(high, low, close)
        if squeeze_on:
            return 0.9 if momentum > 0 else 0.7
        elif momentum > 0:
            return 0.6
        return 0.3

    def calculate_atr(self, prices: List[float]) -> float:
        if len(prices) < 3:
            return 0.0
        rets = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
        return safe_mean(rets[-14:], safe_mean(rets))

    def build_price_ladders(self, prices: List[float], feats: Dict[str, float]) -> Tuple[Dict, Dict, float, float]:
        last = prices[-1]
        window = max(10, len(prices) // 3)
        recent_low = min(prices[-window:])
        recent_high = max(prices[-window:])
        atr_val = self.calculate_atr(prices) or (last * 0.002)
        microtrend = feats.get("microtrend", 0.5)
        momentum = feats.get("momentum_cf", 0.5)
        impulse = feats.get("impulse", 0.5)
        candle = feats.get("candle_proj", 0.5)
        buy_anchor = recent_low + (0.3 * microtrend + 0.2 * momentum) * atr_val
        sell_anchor = recent_high + (0.5 * impulse + 0.3 * candle) * atr_val
        buy_ladder = {}
        sell_ladder = {}
        for t in range(1, self.cfg.ladder_tiers + 1):
            buy_ladder[f"tier{t}"] = round(buy_anchor - (t - 1) * self.cfg.ladder_step_atr * atr_val, 8)
            sell_ladder[f"tier{t}"] = round(sell_anchor + (t - 1) * self.cfg.ladder_sell_step * atr_val, 8)
        return buy_ladder, sell_ladder, round(buy_anchor, 8), round(sell_anchor, 8)

    def determine_stage(self, score: float, feats: Dict[str, float]) -> str:
        if score >= self.cfg.break_out:
            if feats.get("impulse", 0.0) >= 0.8 or feats.get("pressure", 0.0) >= 0.7:
                return "BREAKOUT"
        if score >= self.cfg.break_pre:
            return "PRE_BREAKOUT"
        if score >= 50:
            return "ACCUMULATION"
        return "NEUTRAL"

    async def analyze(self, symbol: str, high: List[float], low: List[float], 
                     close: List[float], volume: List[float]) -> Dict[str, Any]:
        if len(close) < 30:
            return {
                "symbol": symbol,
                "stage": "INSUFFICIENT_DATA",
                "prebreakout_score": 0.0,
                "confidence": 0.0
            }
        liq = [c * v for c, v in zip(close, volume)]
        tasks = await asyncio.gather(
            self.predictive_rsi(close),
            self.momentum_cf(close),
            self.vol_spike(close),
            self.pressure(volume, close),
            self.microtrend(close),
            self.accel(close),
            self.anomaly_vol(volume),
            self.candle_proj(close),
            self.consistency(close),
            self.impulse(close),
            self.liquidity_shift(liq),
            self.squeeze_pressure(high, low, close)
        )
        names = [
            "rsi", "momentum_cf", "vol_spike", "pressure", "microtrend",
            "accel", "anomaly_vol", "candle_proj", "consistency",
            "impulse", "liquidity", "squeeze"
        ]
        feats = {k: round(v, 4) for k, v in zip(names, tasks)}
        raw_score = sum(feats[k] * self.cfg.weights[k] for k in names)
        prebreakout_score = round(clip01(raw_score) * 100.0, 2)
        breakout_prob = round(math.tanh(prebreakout_score / 85.0), 4)
        enhanced_prob = round(1.0 - math.exp(-prebreakout_score / 85.0), 4)
        stage = self.determine_stage(prebreakout_score, feats)
        buy_ladder, sell_ladder, buy_anchor, sell_anchor = self.build_price_ladders(close, feats)
        atr = self.calculate_atr(close)
        stop_loss = close[-1] - (2.0 * atr)
        take_profit = close[-1] + (3.0 * atr)
        reasons = []
        if feats["squeeze"] > 0.7:
            reasons.append("TTM Squeeze Building")
        if feats["impulse"] > 0.6:
            reasons.append("Strong Momentum Impulse")
        if feats["momentum_cf"] > 0.6:
            reasons.append("Bullish Momentum Confirmation")
        if feats["anomaly_vol"] > 0.6:
            reasons.append("Volume Anomaly Detected")
        if feats["microtrend"] > 0.6:
            reasons.append("Strong Micro-Trend")
        if feats["consistency"] > 0.6:
            reasons.append("Consistent Price Action")
        return {
            "symbol": symbol,
            "stage": stage,
            "prebreakout_score": prebreakout_score,
            "breakout_prob": breakout_prob,
            "enhanced_prob": enhanced_prob,
            "current_price": close[-1],
            "buy_anchor": buy_anchor,
            "sell_anchor": sell_anchor,
            "buy_ladder": buy_ladder,
            "sell_ladder": sell_ladder,
            "stop_loss": round(stop_loss, 8),
            "take_profit": round(take_profit, 8),
            "atr": round(atr, 8),
            "features": feats,
            "reasons": reasons
        }


class MultiTimeframeAnalyzer:
    def __init__(self, timeframes: Optional[List[int]] = None):
        self.timeframes = timeframes or [5, 15, 60, 240]
        self.detector = PreBreakoutDetector()

    def fuse_timeframes(self, tf_results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        valid_tfs = [tf for tf in self.timeframes if tf in tf_results]
        if not valid_tfs:
            return {}
        scores = [tf_results[tf]["prebreakout_score"] / 100.0 for tf in valid_tfs]
        alignment = safe_mean([tf_results[tf]["features"]["microtrend"] for tf in valid_tfs])
        momentum = safe_mean([tf_results[tf]["features"]["momentum_cf"] for tf in valid_tfs])
        fusion = clip01(math.tanh(0.4 * safe_mean(scores)))
        final_score = clip01(fusion * (0.7 + 0.3 * ((alignment + momentum) / 2.0)))
        core_tf = 15 if 15 in tf_results else valid_tfs[len(valid_tfs) // 2]
        core = tf_results[core_tf]
        return {
            "prebreakout_score": round(final_score * 100.0, 2),
            "breakout_prob": round(safe_mean([tf_results[tf]["breakout_prob"] for tf in valid_tfs]), 4),
            "enhanced_prob": round(safe_mean([tf_results[tf]["enhanced_prob"] for tf in valid_tfs]), 4),
            "buy_anchor": core["buy_anchor"],
            "sell_anchor": core["sell_anchor"],
            "buy_ladder": core["buy_ladder"],
            "sell_ladder": core["sell_ladder"],
            "stop_loss": core["stop_loss"],
            "take_profit": core["take_profit"],
            "atr": core["atr"],
            "features": core["features"],
            "timeframe_alignment": round(alignment, 4),
            "timeframe_momentum": round(momentum, 4),
            "analyzed_timeframes": valid_tfs
        }
