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
            raw = {
                "range_contraction": 0.065,
                "higher_lows": 0.055,
                "obv_divergence": 0.055,
                "volume_accumulation": 0.055,
                "squeeze": 0.060,
                "volume_profile": 0.055,
                "seller_exhaustion": 0.050,
                "absorption_detection": 0.050,
                "volatility_regime": 0.045,
                "resistance_proximity": 0.040,
                "choppiness": 0.035,
                "consistency": 0.035,
                "relative_volume": 0.035,
                "rsi": 0.030,
                "pressure": 0.025,
                "bid_ask_imbalance": 0.025,
                "order_flow_momentum": 0.025,
                "obv_trend": 0.025,
                "mfi_signal": 0.020,
                "ultimate_osc": 0.020,
                "klinger_signal": 0.020,
                "momentum_cf": 0.015,
                "vol_spike": 0.015,
                "microtrend": 0.015,
                "accel": 0.010,
                "anomaly_vol": 0.010,
                "candle_proj": 0.010,
                "impulse": 0.010,
                "liquidity": 0.010,
                "adx_strength": 0.010,
                "aroon_signal": 0.010,
                "supertrend_conf": 0.010,
                "vortex_signal": 0.010,
                "linreg_trend": 0.010,
                "elder_power": 0.010,
                "donchian_position": 0.010,
                "cci_signal": 0.010,
                "williams_r": 0.010,
                "stoch_rsi": 0.010,
                "pivot_distance": 0.010,
                "fibonacci_level": 0.010,
                "parabolic_sar": 0.010,
                "price_acceleration": 0.010,
                "volume_weighted_momentum": 0.010,
                "market_depth_score": 0.010,
                "trend_strength_index": 0.010,
                "reversal_probability": 0.010,
                "breakout_velocity": 0.005,
                "liquidity_adjusted_momentum": 0.010,
            }
            total = sum(raw.values())
            self.weights = {k: round(v / total, 6) for k, v in raw.items()}


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
        raw_buy_anchor = recent_low + (0.3 * microtrend + 0.2 * momentum) * atr_val
        max_distance = 3.0 * atr_val
        if last - raw_buy_anchor > max_distance:
            buy_anchor = last - max_distance
        else:
            buy_anchor = raw_buy_anchor
        sell_anchor = recent_high + (0.5 * impulse + 0.3 * candle) * atr_val
        if sell_anchor < last + atr_val:
            sell_anchor = last + atr_val
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

    async def williams_r_signal(self, high: List[float], low: List[float], close: List[float]) -> float:
        period = min(14, len(close) - 1)
        if period < 5:
            return 0.5
        highest = max(high[-period:])
        lowest = min(low[-period:])
        if highest - lowest < 1e-9:
            return 0.5
        wr = ((highest - close[-1]) / (highest - lowest)) * -100
        if wr < -80:
            return clip01(0.85 + (abs(wr) - 80) / 80)
        elif wr > -20:
            return clip01(0.25 - (wr + 20) / 80)
        return clip01(0.5 - wr / 200)

    async def stoch_rsi_signal(self, close: List[float]) -> float:
        rsi_vals = []
        period = min(14, len(close) - 1)
        for i in range(period, len(close)):
            rsi_vals.append(self.math.rsi(close[:i+1], period))
        if len(rsi_vals) < 5:
            return 0.5
        stoch_period = min(14, len(rsi_vals))
        lowest_rsi = min(rsi_vals[-stoch_period:])
        highest_rsi = max(rsi_vals[-stoch_period:])
        if highest_rsi - lowest_rsi < 1e-9:
            return 0.5
        stoch_rsi = (rsi_vals[-1] - lowest_rsi) / (highest_rsi - lowest_rsi) * 100
        if stoch_rsi < 20:
            return clip01(0.85 + (20 - stoch_rsi) / 40)
        elif stoch_rsi > 80:
            return clip01(0.25 - (stoch_rsi - 80) / 40)
        return clip01(stoch_rsi / 100)

    async def mfi_signal(self, high: List[float], low: List[float], close: List[float], volume: List[float]) -> float:
        if len(close) < 15:
            return 0.5
        typical = [(h + l + c) / 3 for h, l, c in zip(high, low, close)]
        mf = [t * v for t, v in zip(typical, volume)]
        pos_mf, neg_mf = 0.0, 0.0
        for i in range(1, min(15, len(mf))):
            if typical[-(i)] > typical[-(i+1)]:
                pos_mf += mf[-(i)]
            else:
                neg_mf += mf[-(i)]
        if neg_mf < 1e-9:
            mfi = 100.0
        else:
            mfi = 100 - (100 / (1 + pos_mf / neg_mf))
        if mfi < 20:
            return clip01(0.85 + (20 - mfi) / 40)
        elif mfi > 80:
            return clip01(0.25 - (mfi - 80) / 40)
        return clip01(mfi / 100)

    async def obv_trend_signal(self, close: List[float], volume: List[float]) -> float:
        if len(close) < 20:
            return 0.5
        obv = [0.0]
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv.append(obv[-1] + volume[i])
            elif close[i] < close[i-1]:
                obv.append(obv[-1] - volume[i])
            else:
                obv.append(obv[-1])
        obv_slope = linear_slope(obv[-20:])
        price_slope = linear_slope(close[-20:])
        if obv_slope > 0 and price_slope > 0:
            return clip01(0.7 + tanh01(obv_slope * 1e-6) * 0.25)
        elif obv_slope > 0 and price_slope < 0:
            return clip01(0.75)
        elif obv_slope < 0 and price_slope > 0:
            return clip01(0.35)
        return clip01(0.4 + tanh01(obv_slope * 1e-6) * 0.3)

    async def pivot_distance_signal(self, high: List[float], low: List[float], close: List[float]) -> float:
        if len(close) < 2:
            return 0.5
        pivot = (high[-2] + low[-2] + close[-2]) / 3
        r1 = 2 * pivot - low[-2]
        s1 = 2 * pivot - high[-2]
        current = close[-1]
        if current > r1:
            dist = (current - r1) / (r1 - pivot) if (r1 - pivot) > 1e-9 else 0
            return clip01(0.7 + tanh01(dist) * 0.25)
        elif current < s1:
            dist = (s1 - current) / (pivot - s1) if (pivot - s1) > 1e-9 else 0
            return clip01(0.8 + tanh01(dist) * 0.15)
        position = (current - s1) / (r1 - s1) if (r1 - s1) > 1e-9 else 0.5
        return clip01(position)

    async def fibonacci_level_signal(self, high: List[float], low: List[float], close: List[float]) -> float:
        if len(close) < 20:
            return 0.5
        swing_high = max(high[-20:])
        swing_low = min(low[-20:])
        diff = swing_high - swing_low
        if diff < 1e-9:
            return 0.5
        fib_382 = swing_high - diff * 0.382
        fib_500 = swing_high - diff * 0.500
        fib_618 = swing_high - diff * 0.618
        current = close[-1]
        if current > fib_382:
            return clip01(0.7 + (current - fib_382) / diff * 0.3)
        elif current > fib_500:
            return clip01(0.6)
        elif current > fib_618:
            return clip01(0.7)
        else:
            return clip01(0.8 + (fib_618 - current) / diff * 0.15)

    async def parabolic_sar_signal(self, high: List[float], low: List[float], close: List[float]) -> float:
        if len(close) < 5:
            return 0.5
        af_start, af_step, af_max = 0.02, 0.02, 0.2
        uptrend = close[-1] > close[0]
        sar = low[0] if uptrend else high[0]
        ep = high[0] if uptrend else low[0]
        af = af_start
        for i in range(1, len(close)):
            if uptrend:
                sar = sar + af * (ep - sar)
                if low[i] < sar:
                    uptrend = False
                    sar = ep
                    ep = low[i]
                    af = af_start
                else:
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + af_step, af_max)
            else:
                sar = sar - af * (sar - ep)
                if high[i] > sar:
                    uptrend = True
                    sar = ep
                    ep = high[i]
                    af = af_start
                else:
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + af_step, af_max)
        if uptrend and close[-1] > sar:
            return clip01(0.75 + (close[-1] - sar) / close[-1] * 5)
        elif not uptrend and close[-1] < sar:
            return clip01(0.3 - (sar - close[-1]) / close[-1] * 5)
        return clip01(0.5)

    async def bid_ask_imbalance_signal(self, high: List[float], low: List[float], close: List[float], volume: List[float]) -> float:
        if len(close) < 10:
            return 0.5
        buying_pressure = sum((close[i] - low[i]) * volume[i] for i in range(-10, 0))
        selling_pressure = sum((high[i] - close[i]) * volume[i] for i in range(-10, 0))
        total_pressure = buying_pressure + selling_pressure
        if total_pressure < 1e-9:
            return 0.5
        imbalance = (buying_pressure - selling_pressure) / total_pressure
        return clip01(0.5 + imbalance * 0.5)

    async def order_flow_momentum_signal(self, close: List[float], volume: List[float]) -> float:
        if len(close) < 20:
            return 0.5
        flow = []
        for i in range(1, min(20, len(close))):
            delta = close[-(i)] - close[-(i+1)]
            flow.append(delta * volume[-(i)])
        if not flow:
            return 0.5
        recent_flow = sum(flow[:5])
        older_flow = sum(flow[5:]) if len(flow) > 5 else 0
        if abs(older_flow) < 1e-9:
            return clip01(0.5 + tanh01(recent_flow / 1000) * 0.4)
        flow_acceleration = recent_flow / (abs(older_flow) + 1)
        return clip01(0.5 + tanh01(flow_acceleration) * 0.45)

    async def volatility_regime_signal(self, high: List[float], low: List[float], close: List[float]) -> float:
        if len(close) < 30:
            return 0.5
        atr_recent = safe_mean([high[i] - low[i] for i in range(-10, 0)])
        atr_older = safe_mean([high[i] - low[i] for i in range(-30, -10)])
        if atr_older < 1e-9:
            return 0.5
        volatility_ratio = atr_recent / atr_older
        if volatility_ratio < 0.7:
            return clip01(0.75 + (0.7 - volatility_ratio) * 0.5)
        elif volatility_ratio > 1.5:
            return clip01(0.4 - (volatility_ratio - 1.5) * 0.2)
        return clip01(0.5 + (1 - volatility_ratio) * 0.3)

    async def price_acceleration_signal(self, close: List[float]) -> float:
        if len(close) < 15:
            return 0.5
        velocity_1 = close[-1] - close[-5]
        velocity_2 = close[-5] - close[-10]
        velocity_3 = close[-10] - close[-15] if len(close) >= 15 else velocity_2
        price_norm = close[-1] if close[-1] > 0 else 1.0
        accel_1 = (velocity_1 - velocity_2) / price_norm
        accel_2 = (velocity_2 - velocity_3) / price_norm
        if accel_1 > 0 and accel_2 > 0:
            return clip01(0.7 + tanh01(accel_1 * 50) * 0.25)
        elif accel_1 > 0 and velocity_1 > 0:
            return clip01(0.6 + tanh01(accel_1 * 50) * 0.2)
        elif accel_1 < 0 and velocity_1 < 0:
            return clip01(0.35 - tanh01(abs(accel_1) * 50) * 0.15)
        return clip01(0.5)

    async def volume_weighted_momentum_signal(self, close: List[float], volume: List[float]) -> float:
        if len(close) < 20:
            return 0.5
        vwap_short = sum(c * v for c, v in zip(close[-5:], volume[-5:])) / max(sum(volume[-5:]), 1e-9)
        vwap_long = sum(c * v for c, v in zip(close[-20:], volume[-20:])) / max(sum(volume[-20:]), 1e-9)
        current = close[-1]
        if current > vwap_short > vwap_long:
            return clip01(0.75 + (current - vwap_short) / current * 5)
        elif current > vwap_short:
            return clip01(0.6 + (current - vwap_short) / current * 3)
        elif current < vwap_short < vwap_long:
            return clip01(0.3 - (vwap_short - current) / current * 3)
        return clip01(0.5)

    async def market_depth_score_signal(self, volume: List[float]) -> float:
        if len(volume) < 20:
            return 0.5
        recent_vol = safe_mean(volume[-5:])
        avg_vol = safe_mean(volume[-20:])
        if avg_vol < 1e-9:
            return 0.5
        depth_ratio = recent_vol / avg_vol
        if depth_ratio > 2.0:
            return clip01(0.8 + tanh01((depth_ratio - 2) / 2) * 0.15)
        elif depth_ratio > 1.5:
            return clip01(0.7 + (depth_ratio - 1.5) * 0.2)
        elif depth_ratio < 0.5:
            return clip01(0.35)
        return clip01(0.5 + (depth_ratio - 1) * 0.3)

    async def trend_strength_index_signal(self, close: List[float]) -> float:
        if len(close) < 30:
            return 0.5
        changes = [close[i] - close[i-1] for i in range(-29, 0)]
        up_sum = sum(c for c in changes if c > 0)
        down_sum = abs(sum(c for c in changes if c < 0))
        if up_sum + down_sum < 1e-9:
            return 0.5
        tsi = (up_sum - down_sum) / (up_sum + down_sum)
        return clip01(0.5 + tsi * 0.45)

    async def reversal_probability_signal(self, high: List[float], low: List[float], close: List[float]) -> float:
        if len(close) < 20:
            return 0.5
        swing_high = max(high[-20:])
        swing_low = min(low[-20:])
        range_size = swing_high - swing_low
        if range_size < 1e-9:
            return 0.5
        current_pos = (close[-1] - swing_low) / range_size
        if current_pos < 0.15:
            return clip01(0.85 + (0.15 - current_pos) * 0.5)
        elif current_pos > 0.85:
            return clip01(0.25 - (current_pos - 0.85) * 0.5)
        return clip01(0.5)

    async def breakout_velocity_signal(self, high: List[float], low: List[float], close: List[float]) -> float:
        if len(close) < 20:
            return 0.5
        recent_high = max(high[-5:])
        recent_low = min(low[-5:])
        older_high = max(high[-20:-5]) if len(high) > 5 else recent_high
        older_low = min(low[-20:-5]) if len(low) > 5 else recent_low
        current = close[-1]
        if current > older_high:
            breakout_strength = (current - older_high) / (older_high - older_low + 1e-9)
            return clip01(0.75 + tanh01(breakout_strength * 2) * 0.2)
        elif current < older_low:
            breakdown_strength = (older_low - current) / (older_high - older_low + 1e-9)
            return clip01(0.3 - tanh01(breakdown_strength * 2) * 0.15)
        return clip01(0.5)

    async def liquidity_adjusted_momentum_signal(self, close: List[float], volume: List[float]) -> float:
        if len(close) < 20:
            return 0.5
        momentum = (close[-1] - close[-10]) / close[-10] if close[-10] > 0 else 0
        volume_factor = safe_mean(volume[-5:]) / (safe_mean(volume[-20:]) + 1e-9)
        adjusted_momentum = momentum * math.sqrt(max(0.1, volume_factor))
        return clip01(0.5 + tanh01(adjusted_momentum * 20) * 0.45)

    async def range_contraction_signal(self, high: List[float], low: List[float], close: List[float]) -> float:
        if len(close) < 30:
            return 0.5
        recent_range = max(high[-10:]) - min(low[-10:])
        older_range = max(high[-30:-10]) - min(low[-30:-10])
        if older_range < 1e-9:
            return 0.5
        contraction_ratio = recent_range / older_range
        recent_bars_range = safe_mean([high[i] - low[i] for i in range(-10, 0)])
        older_bars_range = safe_mean([high[i] - low[i] for i in range(-30, -10)])
        bar_contraction = recent_bars_range / (older_bars_range + 1e-9)
        if contraction_ratio < 0.4 and bar_contraction < 0.5:
            return clip01(0.95)
        elif contraction_ratio < 0.55 and bar_contraction < 0.65:
            return clip01(0.85)
        elif contraction_ratio < 0.7:
            return clip01(0.7)
        elif contraction_ratio > 1.5:
            return clip01(0.2)
        return clip01(0.5 + (1.0 - contraction_ratio) * 0.4)

    async def higher_lows_signal(self, high: List[float], low: List[float], close: List[float]) -> float:
        if len(close) < 20:
            return 0.5
        swing_lows = []
        for i in range(2, len(low) - 2):
            if low[i] <= low[i-1] and low[i] <= low[i-2] and low[i] <= low[i+1] and low[i] <= low[i+2]:
                swing_lows.append(low[i])
        recent_lows = swing_lows[-5:] if len(swing_lows) >= 5 else swing_lows[-3:] if len(swing_lows) >= 3 else swing_lows
        if len(recent_lows) < 2:
            low_slope = linear_slope(low[-10:])
            if low_slope > 0:
                return clip01(0.65 + tanh01(low_slope * 20) * 0.15)
            return 0.5
        rising_count = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i] > recent_lows[i-1])
        ratio = rising_count / (len(recent_lows) - 1)
        swing_highs = []
        for i in range(2, len(high) - 2):
            if high[i] >= high[i-1] and high[i] >= high[i-2] and high[i] >= high[i+1] and high[i] >= high[i+2]:
                swing_highs.append(high[i])
        recent_highs = swing_highs[-4:] if len(swing_highs) >= 4 else swing_highs
        highs_flat = False
        if len(recent_highs) >= 2:
            high_range = max(recent_highs) - min(recent_highs)
            avg_high = safe_mean(recent_highs)
            if avg_high > 0 and (high_range / avg_high) < 0.02:
                highs_flat = True
        if ratio >= 0.8 and highs_flat:
            return clip01(0.92)
        elif ratio >= 0.75:
            return clip01(0.8)
        elif ratio >= 0.5:
            return clip01(0.65)
        elif ratio < 0.25:
            return clip01(0.3)
        return clip01(0.5 + ratio * 0.3)

    async def obv_divergence_signal(self, close: List[float], volume: List[float]) -> float:
        if len(close) < 30:
            return 0.5
        obv = [0.0]
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv.append(obv[-1] + volume[i])
            elif close[i] < close[i-1]:
                obv.append(obv[-1] - volume[i])
            else:
                obv.append(obv[-1])
        price_slope = linear_slope(close[-20:])
        obv_slope = linear_slope(obv[-20:])
        price_norm = abs(price_slope) / (close[-1] + 1e-9)
        obv_norm = obv_slope / (abs(safe_mean(obv[-20:])) + 1e-9) if safe_mean(obv[-20:]) != 0 else 0
        if obv_norm > 0.01 and price_norm < 0.001:
            return clip01(0.88)
        elif obv_norm > 0.005 and price_norm < 0.002:
            return clip01(0.78)
        elif obv_norm > 0 and price_slope < 0:
            return clip01(0.82)
        elif obv_norm < -0.005 and price_slope > 0:
            return clip01(0.25)
        return clip01(0.5)

    async def volume_accumulation_signal(self, close: List[float], volume: List[float]) -> float:
        if len(close) < 30:
            return 0.5
        old_avg_vol = safe_mean(volume[-30:-15])
        mid_avg_vol = safe_mean(volume[-15:-5])
        recent_avg_vol = safe_mean(volume[-5:])
        if old_avg_vol < 1e-9:
            return 0.5
        vol_dried = mid_avg_vol / old_avg_vol
        vol_picking_up = recent_avg_vol / (mid_avg_vol + 1e-9)
        price_range = max(close[-15:]) - min(close[-15:])
        price_avg = safe_mean(close[-15:])
        price_flat = (price_range / (price_avg + 1e-9)) < 0.04
        if vol_dried < 0.6 and vol_picking_up > 1.3 and price_flat:
            return clip01(0.92)
        elif vol_dried < 0.7 and vol_picking_up > 1.2 and price_flat:
            return clip01(0.82)
        elif vol_dried < 0.8 and vol_picking_up > 1.1:
            return clip01(0.7)
        elif vol_picking_up > 1.5:
            return clip01(0.65)
        elif vol_dried > 1.3 and vol_picking_up < 0.7:
            return clip01(0.3)
        return clip01(0.5)

    async def resistance_proximity_signal(self, high: List[float], low: List[float], close: List[float]) -> float:
        if len(close) < 30:
            return 0.5
        lookback_high = max(high[-30:])
        current = close[-1]
        price_range = lookback_high - min(low[-30:])
        if price_range < 1e-9:
            return 0.5
        distance_to_resistance = (lookback_high - current) / price_range
        already_broken = current > lookback_high
        if already_broken:
            overshoot = (current - lookback_high) / price_range
            if overshoot > 0.15:
                return clip01(0.15)
            elif overshoot > 0.05:
                return clip01(0.3)
            return clip01(0.45)
        if distance_to_resistance < 0.05:
            return clip01(0.9)
        elif distance_to_resistance < 0.1:
            return clip01(0.8)
        elif distance_to_resistance < 0.2:
            return clip01(0.65)
        elif distance_to_resistance > 0.6:
            return clip01(0.3)
        return clip01(0.5 + (0.3 - distance_to_resistance) * 0.5)

    async def volume_profile_signal(self, high: List[float], low: List[float],
                                     close: List[float], volume: List[float]) -> float:
        if len(close) < 50:
            return 0.5
        price_min = min(low[-50:])
        price_max = max(high[-50:])
        price_range = price_max - price_min
        if price_range < 1e-9:
            return 0.5
        num_bins = 20
        bin_size = price_range / num_bins
        vol_at_price = [0.0] * num_bins
        for i in range(-50, 0):
            bin_idx = int((close[i] - price_min) / bin_size)
            bin_idx = min(bin_idx, num_bins - 1)
            vol_at_price[bin_idx] += volume[i]
        current_bin = int((close[-1] - price_min) / bin_size)
        current_bin = min(current_bin, num_bins - 1)
        total_vol = sum(vol_at_price) or 1.0
        avg_vol_per_bin = total_vol / num_bins
        above_vol = sum(vol_at_price[current_bin + 1:]) if current_bin < num_bins - 1 else 0
        above_bins = max(1, num_bins - current_bin - 1)
        avg_above = above_vol / above_bins
        below_vol = sum(vol_at_price[:current_bin]) if current_bin > 0 else 0
        below_bins = max(1, current_bin)
        avg_below = below_vol / below_bins
        lvn_above = avg_above < avg_vol_per_bin * 0.5
        hvn_below = avg_below > avg_vol_per_bin * 1.2
        at_current = vol_at_price[current_bin]
        building_at_level = at_current > avg_vol_per_bin * 1.5
        if lvn_above and hvn_below and building_at_level:
            return clip01(0.95)
        elif lvn_above and hvn_below:
            return clip01(0.85)
        elif lvn_above and building_at_level:
            return clip01(0.80)
        elif lvn_above:
            return clip01(0.72)
        elif hvn_below:
            return clip01(0.65)
        return clip01(0.5)

    async def seller_exhaustion_signal(self, high: List[float], low: List[float],
                                        close: List[float], volume: List[float]) -> float:
        if len(close) < 30:
            return 0.5
        sell_vol = []
        for i in range(-25, 0):
            if close[i] < close[i - 1]:
                sell_vol.append(volume[i])
            else:
                sell_vol.append(0.0)
        windows = [sell_vol[i:i + 5] for i in range(0, 25, 5)]
        window_sums = [sum(w) for w in windows]
        if len(window_sums) < 3:
            return 0.5
        declining_sell_vol = all(window_sums[i] <= window_sums[i - 1] * 1.05 for i in range(1, len(window_sums)))
        last_3_sell = window_sums[-1]
        first_sell = window_sums[0] if window_sums[0] > 0 else 1.0
        exhaustion_ratio = last_3_sell / first_sell
        price_holding = abs(close[-1] - close[-25]) / (close[-25] + 1e-9) < 0.03
        price_rising = close[-1] > close[-25]
        down_candles_recent = sum(1 for i in range(-5, 0) if close[i] < close[i - 1])
        down_candles_old = sum(1 for i in range(-20, -10) if close[i] < close[i - 1])
        fewer_down = down_candles_recent < down_candles_old
        if declining_sell_vol and exhaustion_ratio < 0.3 and (price_holding or price_rising):
            return clip01(0.93)
        elif exhaustion_ratio < 0.4 and (price_holding or price_rising) and fewer_down:
            return clip01(0.85)
        elif exhaustion_ratio < 0.5 and price_holding:
            return clip01(0.75)
        elif exhaustion_ratio < 0.6:
            return clip01(0.65)
        elif exhaustion_ratio > 1.5:
            return clip01(0.25)
        return clip01(0.5)

    async def absorption_detection_signal(self, high: List[float], low: List[float],
                                           close: List[float], volume: List[float]) -> float:
        if len(close) < 30:
            return 0.5
        support_tests = 0
        support_holds = 0
        support_level = min(low[-20:])
        support_zone = support_level * 1.005
        avg_vol = safe_mean(volume[-30:])
        high_vol_touches = 0
        for i in range(-20, 0):
            if low[i] <= support_zone:
                support_tests += 1
                if close[i] > support_level:
                    support_holds += 1
                if volume[i] > avg_vol * 1.3:
                    high_vol_touches += 1
        resistance_tests = 0
        resistance_level = max(high[-20:])
        resistance_zone = resistance_level * 0.995
        for i in range(-20, 0):
            if high[i] >= resistance_zone:
                resistance_tests += 1
        if support_tests >= 3 and support_holds >= support_tests * 0.8 and high_vol_touches >= 2:
            if resistance_tests <= 1:
                return clip01(0.93)
            return clip01(0.88)
        elif support_tests >= 2 and support_holds >= support_tests * 0.7 and high_vol_touches >= 1:
            return clip01(0.78)
        elif support_tests >= 2 and support_holds >= support_tests * 0.6:
            return clip01(0.68)
        elif support_tests <= 1:
            return clip01(0.5)
        return clip01(0.5 + (support_holds / max(support_tests, 1)) * 0.3)

    async def relative_volume_signal(self, volume: List[float]) -> float:
        if len(volume) < 50:
            return 0.5
        period_len = min(24, len(volume) // 3)
        if period_len < 5:
            return 0.5
        periods = []
        for start in range(0, len(volume) - period_len, period_len):
            periods.append(volume[start:start + period_len])
        if len(periods) < 2:
            return 0.5
        position_in_period = len(volume) % period_len
        if position_in_period == 0:
            position_in_period = period_len
        historical_at_position = []
        for p in periods[:-1]:
            if position_in_period <= len(p):
                window_start = max(0, position_in_period - 3)
                window_end = min(len(p), position_in_period + 2)
                historical_at_position.append(safe_mean(p[window_start:window_end]))
        if not historical_at_position:
            return 0.5
        hist_avg = safe_mean(historical_at_position)
        if hist_avg < 1e-9:
            return 0.5
        current_vol = safe_mean(volume[-3:])
        rvol = current_vol / hist_avg
        if rvol > 3.0:
            return clip01(0.92)
        elif rvol > 2.0:
            return clip01(0.82)
        elif rvol > 1.5:
            return clip01(0.72)
        elif rvol > 1.2:
            return clip01(0.62)
        elif rvol < 0.5:
            return clip01(0.35)
        return clip01(0.5 + (rvol - 1.0) * 0.3)

    async def analyze(self, symbol: str, high: List[float], low: List[float], 
                     close: List[float], volume: List[float],
                     order_book: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
            self.cci_signal(high, low, close),
            self.williams_r_signal(high, low, close),
            self.stoch_rsi_signal(close),
            self.mfi_signal(high, low, close, volume),
            self.obv_trend_signal(close, volume),
            self.pivot_distance_signal(high, low, close),
            self.fibonacci_level_signal(high, low, close),
            self.parabolic_sar_signal(high, low, close),
            self.bid_ask_imbalance_signal(high, low, close, volume),
            self.order_flow_momentum_signal(close, volume),
            self.volatility_regime_signal(high, low, close),
            self.price_acceleration_signal(close),
            self.volume_weighted_momentum_signal(close, volume),
            self.market_depth_score_signal(volume),
            self.trend_strength_index_signal(close),
            self.reversal_probability_signal(high, low, close),
            self.breakout_velocity_signal(high, low, close),
            self.liquidity_adjusted_momentum_signal(close, volume),
            self.range_contraction_signal(high, low, close),
            self.higher_lows_signal(high, low, close),
            self.obv_divergence_signal(close, volume),
            self.volume_accumulation_signal(close, volume),
            self.resistance_proximity_signal(high, low, close),
            self.volume_profile_signal(high, low, close, volume),
            self.seller_exhaustion_signal(high, low, close, volume),
            self.absorption_detection_signal(high, low, close, volume),
            self.relative_volume_signal(volume)
        )
        names = [
            "rsi", "momentum_cf", "vol_spike", "pressure", "microtrend",
            "accel", "anomaly_vol", "candle_proj", "consistency",
            "impulse", "liquidity", "squeeze",
            "adx_strength", "aroon_signal", "supertrend_conf", "vortex_signal",
            "linreg_trend", "elder_power", "ultimate_osc", "choppiness",
            "klinger_signal", "donchian_position", "cci_signal",
            "williams_r", "stoch_rsi", "mfi_signal", "obv_trend",
            "pivot_distance", "fibonacci_level", "parabolic_sar",
            "bid_ask_imbalance", "order_flow_momentum", "volatility_regime",
            "price_acceleration", "volume_weighted_momentum", "market_depth_score",
            "trend_strength_index", "reversal_probability", "breakout_velocity",
            "liquidity_adjusted_momentum",
            "range_contraction", "higher_lows", "obv_divergence",
            "volume_accumulation", "resistance_proximity",
            "volume_profile", "seller_exhaustion", "absorption_detection",
            "relative_volume"
        ]
        feats = {k: round(v, 4) for k, v in zip(names, tasks)}
        if order_book:
            imbalance = order_book.get("imbalance", 0)
            spread_pct = order_book.get("spread_pct", 0)
            bid_depth = order_book.get("bid_depth", 0)
            ask_depth = order_book.get("ask_depth", 0)
            book_score = 0.5
            if imbalance > 0.3:
                book_score = clip01(0.7 + imbalance * 0.3)
            elif imbalance < -0.3:
                book_score = clip01(0.3 + imbalance * 0.3)
            if spread_pct < 0.05:
                book_score = min(1.0, book_score * 1.1)
            if bid_depth > 0 and ask_depth > 0 and bid_depth > ask_depth * 1.5:
                book_score = min(1.0, book_score * 1.15)
            feats["real_book_imbalance"] = round(book_score, 4)
        weights = dict(self.cfg.weights or {})
        if "real_book_imbalance" in feats:
            weights["real_book_imbalance"] = 0.04
        raw_score = sum(feats.get(k, 0) * weights.get(k, 0.0) for k in list(feats.keys()))
        consolidation_signals = sum(1 for k in [
            "squeeze", "range_contraction", "higher_lows", "obv_divergence",
            "volume_accumulation", "resistance_proximity", "volatility_regime",
            "consistency", "choppiness",
            "volume_profile", "seller_exhaustion", "absorption_detection"
        ] if feats.get(k, 0) >= 0.60)
        momentum_signals = sum(1 for k in [
            "momentum_cf", "pressure", "impulse",
            "supertrend_conf", "adx_strength", "linreg_trend",
            "breakout_velocity", "trend_strength_index"
        ] if feats.get(k, 0) >= 0.60)
        if consolidation_signals >= 5:
            raw_score *= self.cfg.momentum_boost * 1.15
        elif consolidation_signals >= 4:
            raw_score *= self.cfg.momentum_boost
        elif consolidation_signals >= 3:
            raw_score *= 1.0 + (self.cfg.momentum_boost - 1.0) * 0.7
        elif consolidation_signals >= 2:
            raw_score *= 1.0 + (self.cfg.momentum_boost - 1.0) * 0.4
        if feats.get("squeeze", 0) >= 0.7 and feats.get("range_contraction", 0) >= 0.65:
            raw_score *= self.cfg.early_detection_bonus
        if feats.get("higher_lows", 0) >= 0.7 and feats.get("resistance_proximity", 0) >= 0.7:
            raw_score *= 1.2
        if feats.get("obv_divergence", 0) >= 0.75 and feats.get("volume_accumulation", 0) >= 0.7:
            raw_score *= 1.18
        if feats.get("choppiness", 0) >= 0.7 and feats.get("volatility_regime", 0) >= 0.65:
            raw_score *= 1.1
        if feats.get("squeeze", 0) >= 0.75 and feats.get("higher_lows", 0) >= 0.6:
            raw_score *= 1.12
        if feats.get("seller_exhaustion", 0) >= 0.7 and feats.get("absorption_detection", 0) >= 0.7:
            raw_score *= 1.20
        if feats.get("volume_profile", 0) >= 0.7 and feats.get("range_contraction", 0) >= 0.65:
            raw_score *= 1.15
        if feats.get("seller_exhaustion", 0) >= 0.7 and feats.get("volume_accumulation", 0) >= 0.7 and feats.get("higher_lows", 0) >= 0.6:
            raw_score *= 1.22
        if feats.get("absorption_detection", 0) >= 0.75 and feats.get("resistance_proximity", 0) >= 0.7 and feats.get("relative_volume", 0) >= 0.6:
            raw_score *= 1.18
        rsi_val = self.math.rsi(close, self.cfg.rsi_period)
        if rsi_val > 75:
            raw_score *= 0.6
        elif rsi_val > 70:
            raw_score *= 0.75
        elif rsi_val > 65:
            raw_score *= 0.85
        if len(close) >= 20:
            recent_move = (close[-1] - min(close[-20:])) / (min(close[-20:]) + 1e-9)
            if recent_move > 0.15:
                raw_score *= 0.5
            elif recent_move > 0.10:
                raw_score *= 0.65
            elif recent_move > 0.07:
                raw_score *= 0.8
        if feats.get("resistance_proximity", 0) < 0.3:
            raw_score *= 0.7
        if momentum_signals >= 6 and consolidation_signals < 2:
            raw_score *= 0.7
        high_value_signals = consolidation_signals + min(momentum_signals, 3)
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
        if feats.get("bid_ask_imbalance", 0) > 0.65:
            reasons.append("Strong Bid-Ask Imbalance (Buyers)")
        if feats.get("order_flow_momentum", 0) > 0.65:
            reasons.append("Positive Order Flow Momentum")
        if feats.get("volatility_regime", 0) > 0.7:
            reasons.append("Low Volatility Regime (Consolidation)")
        if feats.get("price_acceleration", 0) > 0.65:
            reasons.append("Price Acceleration Detected")
        if feats.get("volume_weighted_momentum", 0) > 0.65:
            reasons.append("Volume-Weighted Momentum Bullish")
        if feats.get("market_depth_score", 0) > 0.7:
            reasons.append("Strong Market Depth")
        if feats.get("trend_strength_index", 0) > 0.65:
            reasons.append("High Trend Strength Index")
        if feats.get("reversal_probability", 0) > 0.75:
            reasons.append("High Reversal Probability (Oversold)")
        if feats.get("breakout_velocity", 0) > 0.7:
            reasons.append("High Breakout Velocity")
        if feats.get("liquidity_adjusted_momentum", 0) > 0.65:
            reasons.append("Liquidity-Adjusted Momentum Signal")
        if feats.get("range_contraction", 0) > 0.7:
            reasons.append("Range Contraction (Coiling)")
        if feats.get("higher_lows", 0) > 0.7:
            reasons.append("Higher Lows Pattern (Ascending Support)")
        if feats.get("obv_divergence", 0) > 0.7:
            reasons.append("OBV Divergence (Smart Money Accumulation)")
        if feats.get("volume_accumulation", 0) > 0.7:
            reasons.append("Volume Accumulation Pattern")
        if feats.get("resistance_proximity", 0) > 0.75:
            reasons.append("Near Resistance (Breakout Zone)")
        if feats.get("volume_profile", 0) > 0.7:
            reasons.append("Volume Profile: Low Resistance Above (Clear Path)")
        if feats.get("seller_exhaustion", 0) > 0.7:
            reasons.append("Seller Exhaustion: Selling Pressure Drying Up")
        if feats.get("absorption_detection", 0) > 0.7:
            reasons.append("Absorption: Large Buyers Defending Support")
        if feats.get("relative_volume", 0) > 0.7:
            reasons.append("Relative Volume Surge (vs Historical)")
        if feats.get("real_book_imbalance", 0) > 0.7:
            reasons.append("Order Book: Strong Buy-Side Imbalance (Live)")
        if rsi_val > 70:
            reasons.append(f"WARNING: RSI Overbought ({rsi_val:.0f}) - Already Extended")
        if len(close) >= 20:
            recent_move_check = (close[-1] - min(close[-20:])) / (min(close[-20:]) + 1e-9)
            if recent_move_check > 0.10:
                reasons.append(f"WARNING: Already moved +{recent_move_check*100:.1f}% recently")
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
