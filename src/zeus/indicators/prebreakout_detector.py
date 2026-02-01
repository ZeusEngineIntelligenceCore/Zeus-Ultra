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
    spike_cap: float = 0.20
    pressure_cap: float = 1.5
    impulse_cap: float = 5.0
    liquidity_scale: float = 0.5
    break_pre: float = 60.0
    break_out: float = 78.0
    ladder_tiers: int = 5
    ladder_step_atr: float = 0.4
    ladder_sell_step: float = 0.6
    min_confluence_score: float = 0.55
    momentum_boost: float = 1.4
    volume_confirmation_mult: float = 1.6
    early_detection_bonus: float = 1.25
    weights: Optional[Dict[str, float]] = None

    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                "rsi": 0.06,
                "momentum_cf": 0.08,
                "vol_spike": 0.06,
                "pressure": 0.07,
                "microtrend": 0.06,
                "accel": 0.04,
                "anomaly_vol": 0.04,
                "candle_proj": 0.03,
                "consistency": 0.03,
                "impulse": 0.04,
                "liquidity": 0.02,
                "squeeze": 0.02,
                "adx_strength": 0.05,
                "aroon_signal": 0.04,
                "supertrend_conf": 0.05,
                "vortex_signal": 0.04,
                "linreg_trend": 0.05,
                "elder_power": 0.04,
                "ultimate_osc": 0.04,
                "choppiness": 0.03,
                "klinger_signal": 0.04,
                "donchian_position": 0.04,
                "cci_signal": 0.03
            }


class PreBreakoutDetector:
    def __init__(self, config: Optional[BreakoutConfig] = None):
        self.cfg = config or BreakoutConfig()
        self.math = MathKernel()

    async def predictive_rsi(self, prices: List[float]) -> float:
        if len(prices) < 4:
            return 0.5
        rsi = self.math.rsi(prices, self.cfg.rsi_period)
        if rsi < 25:
            return clip01(0.9 + (25 - rsi) / 50)
        elif rsi < 35:
            return clip01(0.7 + (35 - rsi) / 50)
        elif rsi > 75:
            return clip01(0.3 - (rsi - 75) / 50)
        elif rsi > 65:
            return clip01(0.5 - (rsi - 65) / 50)
        normalized = (rsi - 30) / 40
        return clip01(normalized)

    async def momentum_cf(self, prices: List[float]) -> float:
        if len(prices) < 4:
            return 0.5
        slope_short = linear_slope(prices[-8:])
        slope_mid = linear_slope(prices[-15:])
        slope_long = linear_slope(prices[-25:]) if len(prices) >= 25 else slope_mid
        if slope_short > 0 and slope_mid > 0 and slope_long > 0:
            alignment_bonus = 0.15
        elif slope_short < 0 and slope_mid < 0 and slope_long < 0:
            alignment_bonus = 0.15
        else:
            alignment_bonus = 0.0
        combined_slope = slope_short * 0.5 + slope_mid * 0.3 + slope_long * 0.2
        acceleration = slope_short - slope_mid
        accel_factor = clip01(tanh01(acceleration * 10))
        base = clip01(tanh01(combined_slope * 8))
        return clip01(base * 0.7 + accel_factor * 0.15 + alignment_bonus)

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
        vol_trend = safe_mean(vols[-5:]) / (safe_mean(vols[-20:]) or 1e-9) if len(vols) >= 20 else 1.0
        price_momentum = safe_mean(rets[-3:]) if len(rets) >= 3 else last
        base_pressure = abs(last) * vol_ratio
        trend_factor = clip01(vol_trend / 2.0)
        momentum_factor = clip01(tanh01(price_momentum * 15))
        combined = base_pressure * 0.5 + trend_factor * 0.25 + momentum_factor * 0.25
        return clip01(combined / self.cfg.pressure_cap * self.cfg.volume_confirmation_mult)

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
        impulse = feats.get("impulse", 0.0)
        pressure = feats.get("pressure", 0.0)
        momentum = feats.get("momentum_cf", 0.0)
        squeeze = feats.get("squeeze", 0.0)
        vol_spike = feats.get("vol_spike", 0.0)
        confluence_count = sum(1 for v in [impulse, pressure, momentum, squeeze] if v >= self.cfg.min_confluence_score)
        if score >= self.cfg.break_out:
            if impulse >= 0.75 or pressure >= 0.65 or confluence_count >= 3:
                return "BREAKOUT"
            elif vol_spike >= 0.7 and momentum >= 0.6:
                return "BREAKOUT"
        if score >= self.cfg.break_pre:
            if confluence_count >= 2 or (squeeze >= 0.7 and momentum >= 0.5):
                return "PRE_BREAKOUT"
            elif score >= self.cfg.break_pre + 5:
                return "PRE_BREAKOUT"
            return "ACCUMULATION"
        if score >= 45:
            if momentum >= 0.55 or squeeze >= 0.65:
                return "ACCUMULATION"
        if score >= 35:
            return "BUILDING"
        return "NEUTRAL"

    async def adx_strength(self, high: List[float], low: List[float], close: List[float]) -> float:
        adx, plus_di, minus_di = self.math.adx(high, low, close)
        if adx > 25 and plus_di > minus_di:
            return clip01(0.7 + (adx - 25) / 50)
        elif adx > 25 and minus_di > plus_di:
            return clip01(0.3 - (adx - 25) / 100)
        return clip01(adx / 50)

    async def aroon_signal(self, high: List[float], low: List[float]) -> float:
        aroon_up, aroon_down, aroon_osc = self.math.aroon(high, low)
        if aroon_up > 70 and aroon_down < 30:
            return clip01(0.8 + aroon_osc / 200)
        elif aroon_down > 70 and aroon_up < 30:
            return clip01(0.2 - aroon_osc / 200)
        return clip01(0.5 + aroon_osc / 200)

    async def supertrend_conf(self, high: List[float], low: List[float], close: List[float]) -> float:
        st_val, is_uptrend = self.math.supertrend(high, low, close)
        distance = abs(close[-1] - st_val) / (close[-1] + 1e-9)
        if is_uptrend:
            return clip01(0.7 + distance * 5)
        return clip01(0.3 - distance * 5)

    async def vortex_signal(self, high: List[float], low: List[float], close: List[float]) -> float:
        vi_plus, vi_minus = self.math.vortex_indicator(high, low, close)
        diff = vi_plus - vi_minus
        if vi_plus > vi_minus:
            return clip01(0.6 + diff * 2)
        return clip01(0.4 + diff * 2)

    async def linreg_trend(self, close: List[float]) -> float:
        val, slope, r2 = self.math.linear_regression(close, 20)
        if slope > 0 and r2 > 0.7:
            return clip01(0.8 + r2 * 0.2)
        elif slope > 0:
            return clip01(0.5 + slope * 10)
        elif slope < 0 and r2 > 0.7:
            return clip01(0.2 - r2 * 0.2)
        return clip01(0.5 + slope * 5)

    async def elder_power(self, high: List[float], low: List[float], close: List[float]) -> float:
        bull, bear = self.math.elder_ray(high, low, close)
        net_power = bull + bear
        if bull > 0 and bear > 0:
            return clip01(0.8)
        elif bull > 0 and bear < 0 and bull > abs(bear):
            return clip01(0.7 + tanh01(net_power * 5) * 0.2)
        elif bear < 0 and bull < abs(bear):
            return clip01(0.3 - tanh01(abs(net_power) * 5) * 0.2)
        return clip01(0.5 + tanh01(net_power * 5) * 0.3)

    async def ultimate_osc(self, high: List[float], low: List[float], close: List[float]) -> float:
        uo = self.math.ultimate_oscillator(high, low, close)
        if uo < 30:
            return clip01(0.8 + (30 - uo) / 60)
        elif uo > 70:
            return clip01(0.3 - (uo - 70) / 60)
        return clip01((uo - 30) / 40)

    async def choppiness_signal(self, high: List[float], low: List[float], close: List[float]) -> float:
        chop = self.math.choppiness_index(high, low, close)
        if chop < 38.2:
            return clip01(0.85)
        elif chop > 61.8:
            return clip01(0.25)
        return clip01(1.0 - chop / 100)

    async def klinger_signal_score(self, high: List[float], low: List[float], close: List[float], volume: List[float]) -> float:
        kvo, signal = self.math.klinger_oscillator(high, low, close, volume)
        if kvo > signal and kvo > 0:
            return clip01(0.75 + tanh01(kvo / 1000000) * 0.2)
        elif kvo > signal:
            return clip01(0.6)
        elif kvo < signal and kvo < 0:
            return clip01(0.25 - tanh01(abs(kvo) / 1000000) * 0.2)
        return clip01(0.4)

    async def donchian_position(self, high: List[float], low: List[float], close: List[float]) -> float:
        upper, mid, lower = self.math.donchian_channels(high, low)
        if upper - lower < 1e-9:
            return 0.5
        position = (close[-1] - lower) / (upper - lower)
        if position > 0.8:
            return clip01(0.7 + (position - 0.8) * 1.5)
        elif position < 0.2:
            return clip01(0.8 + (0.2 - position) * 1.0)
        return clip01(position)

    async def cci_signal(self, high: List[float], low: List[float], close: List[float]) -> float:
        cci = self.math.cci(high, low, close)
        if cci < -100:
            return clip01(0.85 + (abs(cci) - 100) / 400)
        elif cci > 100:
            return clip01(0.25 - (cci - 100) / 400)
        return clip01(0.5 + cci / 400)

    async def analyze(self, symbol: str, high: List[float], low: List[float], 
                     close: List[float], volume: List[float]) -> Dict[str, Any]:
        if len(close) < 50:
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
            self.squeeze_pressure(high, low, close),
            self.adx_strength(high, low, close),
            self.aroon_signal(high, low),
            self.supertrend_conf(high, low, close),
            self.vortex_signal(high, low, close),
            self.linreg_trend(close),
            self.elder_power(high, low, close),
            self.ultimate_osc(high, low, close),
            self.choppiness_signal(high, low, close),
            self.klinger_signal_score(high, low, close, volume),
            self.donchian_position(high, low, close),
            self.cci_signal(high, low, close)
        )
        names = [
            "rsi", "momentum_cf", "vol_spike", "pressure", "microtrend",
            "accel", "anomaly_vol", "candle_proj", "consistency",
            "impulse", "liquidity", "squeeze",
            "adx_strength", "aroon_signal", "supertrend_conf", "vortex_signal",
            "linreg_trend", "elder_power", "ultimate_osc", "choppiness",
            "klinger_signal", "donchian_position", "cci_signal"
        ]
        feats = {k: round(v, 4) for k, v in zip(names, tasks)}
        weights = self.cfg.weights or {}
        raw_score = sum(feats[k] * weights.get(k, 0.0) for k in names)
        high_value_signals = sum(1 for k in [
            "momentum_cf", "pressure", "impulse", "squeeze", 
            "supertrend_conf", "adx_strength", "linreg_trend"
        ] if feats.get(k, 0) >= 0.60)
        if high_value_signals >= 5:
            raw_score *= self.cfg.momentum_boost * 1.1
        elif high_value_signals >= 4:
            raw_score *= self.cfg.momentum_boost
        elif high_value_signals >= 3:
            raw_score *= 1.0 + (self.cfg.momentum_boost - 1.0) * 0.7
        elif high_value_signals >= 2:
            raw_score *= 1.0 + (self.cfg.momentum_boost - 1.0) * 0.4
        if feats["vol_spike"] >= 0.65 and feats["momentum_cf"] >= 0.55:
            raw_score *= 1.18
        if feats["squeeze"] >= 0.75 and feats["microtrend"] >= 0.55:
            raw_score *= self.cfg.early_detection_bonus
        if feats["accel"] >= 0.65 and feats["pressure"] >= 0.6:
            raw_score *= 1.12
        if feats.get("supertrend_conf", 0) >= 0.7 and feats.get("vortex_signal", 0) >= 0.6:
            raw_score *= 1.15
        if feats.get("aroon_signal", 0) >= 0.75 and feats.get("donchian_position", 0) >= 0.7:
            raw_score *= 1.12
        if feats.get("choppiness", 0) >= 0.7:
            raw_score *= 1.08
        prebreakout_score = round(clip01(raw_score) * 100.0, 2)
        breakout_prob = round(math.tanh(prebreakout_score / 80.0), 4)
        enhanced_prob = round(1.0 - math.exp(-prebreakout_score / 75.0), 4)
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
        if feats.get("supertrend_conf", 0) > 0.7:
            reasons.append("Supertrend Bullish Confirmation")
        if feats.get("adx_strength", 0) > 0.65:
            reasons.append("Strong ADX Trend")
        if feats.get("aroon_signal", 0) > 0.75:
            reasons.append("Aroon Breakout Signal")
        if feats.get("vortex_signal", 0) > 0.65:
            reasons.append("Vortex Bullish Crossover")
        if feats.get("linreg_trend", 0) > 0.7:
            reasons.append("Strong Linear Trend")
        if feats.get("choppiness", 0) > 0.7:
            reasons.append("Low Choppiness - Trending")
        if feats.get("elder_power", 0) > 0.7:
            reasons.append("Elder Ray Bull Power")
        if feats.get("klinger_signal", 0) > 0.7:
            reasons.append("Klinger Volume Confirmation")
        confidence = min(100, max(0, prebreakout_score * 0.9 + high_value_signals * 2))
        return {
            "symbol": symbol,
            "stage": stage,
            "prebreakout_score": prebreakout_score,
            "breakout_prob": breakout_prob,
            "enhanced_prob": enhanced_prob,
            "confidence": round(confidence, 2),
            "current_price": close[-1],
            "buy_anchor": buy_anchor,
            "sell_anchor": sell_anchor,
            "buy_ladder": buy_ladder,
            "sell_ladder": sell_ladder,
            "stop_loss": round(stop_loss, 8),
            "take_profit": round(take_profit, 8),
            "atr": round(atr, 8),
            "features": feats,
            "reasons": reasons,
            "kpi_count": len(names),
            "high_value_signals": high_value_signals
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
