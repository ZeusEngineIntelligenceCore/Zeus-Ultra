#!/usr/bin/env python3
"""
ZEUS MATH KERNEL - Advanced Technical Analysis Engine
Implements 25+ indicators with optimized calculations
"""

from __future__ import annotations
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


class MathKernel:
    EPS = 1e-12

    @staticmethod
    def sma(vals: List[float], length: int) -> float:
        if not vals or length <= 0:
            return 0.0
        n = min(length, len(vals))
        return sum(vals[-n:]) / float(n)

    @staticmethod
    def ema(vals: List[float], length: int) -> float:
        if not vals:
            return 0.0
        n = min(length, len(vals))
        if n <= 1:
            return vals[-1]
        k = 2.0 / (length + 1.0)
        e = vals[-n]
        for v in vals[-n + 1:]:
            e = e + k * (v - e)
        return e

    @staticmethod
    def wma(vals: List[float], length: int) -> float:
        if not vals or length <= 0:
            return 0.0
        n = min(length, len(vals))
        weights = list(range(1, n + 1))
        total_weight = sum(weights)
        weighted_sum = sum(w * v for w, v in zip(weights, vals[-n:]))
        return weighted_sum / total_weight

    @staticmethod
    def hull_ma(vals: List[float], length: int) -> float:
        if len(vals) < length:
            return MathKernel.ema(vals, length)
        half_length = length // 2
        sqrt_length = int(math.sqrt(length))
        wma_half = MathKernel.wma(vals, half_length)
        wma_full = MathKernel.wma(vals, length)
        raw_hull = 2 * wma_half - wma_full
        return raw_hull

    @staticmethod
    def tr_series(high: List[float], low: List[float], close: List[float]) -> List[float]:
        n = min(len(high), len(low), len(close))
        out = []
        for i in range(1, n):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1])
            )
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
        if avg_loss < cls.EPS:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @classmethod
    def stochastic_rsi(cls, close: List[float], rsi_length: int = 14, stoch_length: int = 14) -> Tuple[float, float]:
        if len(close) < rsi_length + stoch_length:
            return 50.0, 50.0
        rsi_values = []
        for i in range(stoch_length + 5, len(close) + 1):
            rsi_values.append(cls.rsi(close[:i], rsi_length))
        if len(rsi_values) < stoch_length:
            return 50.0, 50.0
        recent_rsi = rsi_values[-stoch_length:]
        min_rsi = min(recent_rsi)
        max_rsi = max(recent_rsi)
        if max_rsi - min_rsi < cls.EPS:
            k = 50.0
        else:
            k = ((rsi_values[-1] - min_rsi) / (max_rsi - min_rsi)) * 100
        d = cls.sma(rsi_values[-3:], 3)
        return k, d

    @classmethod
    def macd(cls, close: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        if len(close) < slow:
            return 0.0, 0.0, 0.0
        fast_ema = cls.ema(close, fast)
        slow_ema = cls.ema(close, slow)
        macd_line = fast_ema - slow_ema
        macd_history = []
        for i in range(slow, len(close) + 1):
            f = cls.ema(close[:i], fast)
            s = cls.ema(close[:i], slow)
            macd_history.append(f - s)
        signal_line = cls.ema(macd_history, signal) if len(macd_history) >= signal else macd_line
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @classmethod
    def bollinger_bands(cls, close: List[float], length: int = 20, std_dev: float = 2.0) -> Tuple[float, float, float]:
        if len(close) < length:
            if close:
                return close[-1], close[-1], close[-1]
            return 0.0, 0.0, 0.0
        recent = close[-length:]
        mid = sum(recent) / length
        variance = sum((x - mid) ** 2 for x in recent) / length
        std = math.sqrt(variance)
        upper = mid + (std_dev * std)
        lower = mid - (std_dev * std)
        return upper, mid, lower

    @classmethod
    def keltner_channels(cls, high: List[float], low: List[float], close: List[float], 
                         length: int = 20, mult: float = 2.0) -> Tuple[float, float, float]:
        if len(close) < length:
            if close:
                return close[-1], close[-1], close[-1]
            return 0.0, 0.0, 0.0
        mid = cls.ema(close, length)
        atr_val = cls.atr(high, low, close, length)
        upper = mid + (mult * atr_val)
        lower = mid - (mult * atr_val)
        return upper, mid, lower

    @classmethod
    def ttm_squeeze(cls, high: List[float], low: List[float], close: List[float], 
                    length: int = 20) -> Tuple[bool, float, float]:
        if len(close) < length + 2:
            return False, 0.0, 0.0
        bb_upper, bb_mid, bb_lower = cls.bollinger_bands(close, length, 2.0)
        kc_upper, kc_mid, kc_lower = cls.keltner_channels(high, low, close, length, 1.5)
        squeeze_on = bb_lower > kc_lower and bb_upper < kc_upper
        momentum = close[-1] - cls.sma(close, length)
        return squeeze_on, momentum, bb_upper - bb_lower

    @classmethod
    def adx(cls, high: List[float], low: List[float], close: List[float], length: int = 14) -> Tuple[float, float, float]:
        if len(high) < length + 2:
            return 0.0, 0.0, 0.0
        plus_dm_list = []
        minus_dm_list = []
        tr_list = []
        for i in range(1, len(high)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            plus_dm = up_move if up_move > down_move and up_move > 0 else 0
            minus_dm = down_move if down_move > up_move and down_move > 0 else 0
            plus_dm_list.append(plus_dm)
            minus_dm_list.append(minus_dm)
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            tr_list.append(tr)
        smoothed_plus_dm = cls.ema(plus_dm_list, length)
        smoothed_minus_dm = cls.ema(minus_dm_list, length)
        smoothed_tr = cls.ema(tr_list, length)
        if smoothed_tr < cls.EPS:
            return 0.0, 0.0, 0.0
        plus_di = 100 * (smoothed_plus_dm / smoothed_tr)
        minus_di = 100 * (smoothed_minus_dm / smoothed_tr)
        dx_sum = plus_di + minus_di
        if dx_sum < cls.EPS:
            dx = 0.0
        else:
            dx = 100 * abs(plus_di - minus_di) / dx_sum
        return dx, plus_di, minus_di

    @classmethod
    def obv(cls, close: List[float], volume: List[float]) -> float:
        if len(close) < 2 or len(volume) < 2:
            return 0.0
        obv_val = 0.0
        n = min(len(close), len(volume))
        for i in range(1, n):
            if close[i] > close[i-1]:
                obv_val += volume[i]
            elif close[i] < close[i-1]:
                obv_val -= volume[i]
        return obv_val

    @classmethod
    def vwap(cls, high: List[float], low: List[float], close: List[float], 
             volume: List[float], length: int = 20) -> float:
        n = min(len(high), len(low), len(close), len(volume), length)
        if n < 1:
            return 0.0
        typical_prices = [(high[-i] + low[-i] + close[-i]) / 3 for i in range(1, n + 1)]
        volumes = [volume[-i] for i in range(1, n + 1)]
        total_pv = sum(tp * v for tp, v in zip(typical_prices, volumes))
        total_vol = sum(volumes)
        if total_vol < cls.EPS:
            return close[-1] if close else 0.0
        return total_pv / total_vol

    @classmethod
    def mfi(cls, high: List[float], low: List[float], close: List[float], 
            volume: List[float], length: int = 14) -> float:
        n = min(len(high), len(low), len(close), len(volume))
        if n < length + 1:
            return 50.0
        typical_prices = [(high[i] + low[i] + close[i]) / 3 for i in range(n)]
        raw_money_flow = [typical_prices[i] * volume[i] for i in range(n)]
        positive_flow = 0.0
        negative_flow = 0.0
        for i in range(-length, 0):
            if typical_prices[i] > typical_prices[i-1]:
                positive_flow += raw_money_flow[i]
            elif typical_prices[i] < typical_prices[i-1]:
                negative_flow += raw_money_flow[i]
        if negative_flow < cls.EPS:
            return 100.0
        money_ratio = positive_flow / negative_flow
        mfi = 100 - (100 / (1 + money_ratio))
        return mfi

    @classmethod
    def cci(cls, high: List[float], low: List[float], close: List[float], length: int = 20) -> float:
        n = min(len(high), len(low), len(close))
        if n < length:
            return 0.0
        typical_prices = [(high[i] + low[i] + close[i]) / 3 for i in range(-length, 0)]
        mean_tp = sum(typical_prices) / length
        mean_dev = sum(abs(tp - mean_tp) for tp in typical_prices) / length
        if mean_dev < cls.EPS:
            return 0.0
        cci = (typical_prices[-1] - mean_tp) / (0.015 * mean_dev)
        return cci

    @classmethod
    def williams_r(cls, high: List[float], low: List[float], close: List[float], length: int = 14) -> float:
        n = min(len(high), len(low), len(close))
        if n < length:
            return -50.0
        highest_high = max(high[-length:])
        lowest_low = min(low[-length:])
        if highest_high - lowest_low < cls.EPS:
            return -50.0
        wr = ((highest_high - close[-1]) / (highest_high - lowest_low)) * -100
        return wr

    @classmethod
    def ichimoku(cls, high: List[float], low: List[float], close: List[float]) -> dict:
        def donchian_mid(h, l, length):
            n = min(len(h), len(l), length)
            if n < 1:
                return 0.0
            return (max(h[-n:]) + min(l[-n:])) / 2
        tenkan = donchian_mid(high, low, 9)
        kijun = donchian_mid(high, low, 26)
        senkou_a = (tenkan + kijun) / 2
        senkou_b = donchian_mid(high, low, 52)
        chikou = close[-1] if close else 0.0
        return {
            "tenkan": tenkan,
            "kijun": kijun,
            "senkou_a": senkou_a,
            "senkou_b": senkou_b,
            "chikou": chikou
        }

    @classmethod
    def pivot_points(cls, high: float, low: float, close: float) -> dict:
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)
        return {
            "pivot": pivot,
            "r1": r1, "r2": r2, "r3": r3,
            "s1": s1, "s2": s2, "s3": s3
        }

    @classmethod
    def fibonacci_levels(cls, high: float, low: float, is_uptrend: bool = True) -> dict:
        diff = high - low
        levels = {}
        fib_ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618]
        for ratio in fib_ratios:
            if is_uptrend:
                levels[f"fib_{ratio}"] = high - (diff * ratio)
            else:
                levels[f"fib_{ratio}"] = low + (diff * ratio)
        return levels

    @classmethod
    def volume_profile(cls, close: List[float], volume: List[float], bins: int = 20) -> dict:
        if len(close) < 2 or len(volume) < 2:
            return {"poc": 0.0, "vah": 0.0, "val": 0.0}
        min_price = min(close)
        max_price = max(close)
        if max_price - min_price < cls.EPS:
            return {"poc": close[-1], "vah": close[-1], "val": close[-1]}
        bin_size = (max_price - min_price) / bins
        volume_at_price = [0.0] * bins
        for i, (p, v) in enumerate(zip(close, volume)):
            bin_idx = int((p - min_price) / bin_size)
            bin_idx = min(bin_idx, bins - 1)
            volume_at_price[bin_idx] += v
        poc_idx = volume_at_price.index(max(volume_at_price))
        poc = min_price + (poc_idx + 0.5) * bin_size
        total_vol = sum(volume_at_price)
        cumulative = 0.0
        val_idx = 0
        vah_idx = bins - 1
        for i, v in enumerate(volume_at_price):
            cumulative += v
            if cumulative >= total_vol * 0.3 and val_idx == 0:
                val_idx = i
            if cumulative >= total_vol * 0.7:
                vah_idx = i
                break
        val = min_price + (val_idx + 0.5) * bin_size
        vah = min_price + (vah_idx + 0.5) * bin_size
        return {"poc": poc, "vah": vah, "val": val}

    @classmethod
    def breakout_energy(cls, high: List[float], low: List[float], close: List[float], 
                        volume: List[float], window: int = 20) -> float:
        if len(close) < window + 2:
            return 0.0
        curr_atr = cls.atr(high, low, close, window)
        trs = cls.tr_series(high, low, close)
        atr_base = cls.sma(trs, window) + cls.EPS
        atr_ratio = curr_atr / atr_base
        vol_ratio = volume[-1] / (cls.sma(volume, window) + cls.EPS)
        trend = 1.0 if close[-1] > cls.ema(close, window) else 0.5
        bei = atr_ratio * vol_ratio * trend
        return max(0.0, min(100.0, 20.0 * bei))

    @classmethod
    def momentum_oscillator(cls, close: List[float], length: int = 10) -> float:
        if len(close) < length + 1:
            return 0.0
        base_price = close[-length - 1]
        if abs(base_price) < cls.EPS:
            return 0.0
        return ((close[-1] / base_price) - 1) * 100

    @classmethod
    def rate_of_change(cls, close: List[float], length: int = 12) -> float:
        if len(close) < length + 1:
            return 0.0
        base_price = close[-length - 1]
        if abs(base_price) < cls.EPS:
            return 0.0
        return ((close[-1] - base_price) / base_price) * 100

    @classmethod
    def chaikin_money_flow(cls, high: List[float], low: List[float], close: List[float], 
                           volume: List[float], length: int = 20) -> float:
        n = min(len(high), len(low), len(close), len(volume))
        if n < length:
            return 0.0
        mf_volume = 0.0
        total_volume = 0.0
        for i in range(-length, 0):
            hl_range = high[i] - low[i]
            if hl_range > cls.EPS:
                mf_multiplier = ((close[i] - low[i]) - (high[i] - close[i])) / hl_range
            else:
                mf_multiplier = 0.0
            mf_volume += mf_multiplier * volume[i]
            total_volume += volume[i]
        if total_volume < cls.EPS:
            return 0.0
        return mf_volume / total_volume

    @classmethod
    def dema(cls, vals: List[float], length: int) -> float:
        if len(vals) < length:
            return cls.ema(vals, length)
        ema1 = cls.ema(vals, length)
        ema_series = []
        for i in range(length, len(vals) + 1):
            ema_series.append(cls.ema(vals[:i], length))
        ema2 = cls.ema(ema_series, length) if len(ema_series) >= length else ema1
        return 2 * ema1 - ema2

    @classmethod
    def tema(cls, vals: List[float], length: int) -> float:
        if len(vals) < length:
            return cls.ema(vals, length)
        ema1 = cls.ema(vals, length)
        ema_series = []
        for i in range(length, len(vals) + 1):
            ema_series.append(cls.ema(vals[:i], length))
        ema2 = cls.ema(ema_series, length) if len(ema_series) >= length else ema1
        ema2_series = []
        for i in range(length, len(ema_series) + 1):
            ema2_series.append(cls.ema(ema_series[:i], length))
        ema3 = cls.ema(ema2_series, length) if len(ema2_series) >= length else ema2
        return 3 * ema1 - 3 * ema2 + ema3

    @classmethod
    def trix(cls, close: List[float], length: int = 18) -> float:
        if len(close) < length * 3:
            return 0.0
        ema1 = [cls.ema(close[:i+1], length) for i in range(len(close))]
        ema2 = [cls.ema(ema1[:i+1], length) for i in range(len(ema1))]
        ema3 = [cls.ema(ema2[:i+1], length) for i in range(len(ema2))]
        if len(ema3) < 2 or abs(ema3[-2]) < cls.EPS:
            return 0.0
        return ((ema3[-1] - ema3[-2]) / ema3[-2]) * 10000

    @classmethod
    def ultimate_oscillator(cls, high: List[float], low: List[float], close: List[float],
                            p1: int = 7, p2: int = 14, p3: int = 28) -> float:
        n = min(len(high), len(low), len(close))
        if n < p3 + 1:
            return 50.0
        bp_list = []
        tr_list = []
        for i in range(1, n):
            true_low = min(low[i], close[i-1])
            bp = close[i] - true_low
            tr = max(high[i], close[i-1]) - true_low
            bp_list.append(bp)
            tr_list.append(tr)
        if len(bp_list) < p3:
            return 50.0
        avg1 = sum(bp_list[-p1:]) / (sum(tr_list[-p1:]) + cls.EPS)
        avg2 = sum(bp_list[-p2:]) / (sum(tr_list[-p2:]) + cls.EPS)
        avg3 = sum(bp_list[-p3:]) / (sum(tr_list[-p3:]) + cls.EPS)
        uo = 100 * ((4 * avg1) + (2 * avg2) + avg3) / 7
        return uo

    @classmethod
    def aroon(cls, high: List[float], low: List[float], length: int = 25) -> Tuple[float, float, float]:
        n = min(len(high), len(low))
        if n < length:
            return 50.0, 50.0, 0.0
        recent_high = high[-length:]
        recent_low = low[-length:]
        high_idx = recent_high.index(max(recent_high))
        low_idx = recent_low.index(min(recent_low))
        aroon_up = ((length - (length - 1 - high_idx)) / length) * 100
        aroon_down = ((length - (length - 1 - low_idx)) / length) * 100
        aroon_osc = aroon_up - aroon_down
        return aroon_up, aroon_down, aroon_osc

    @classmethod
    def parabolic_sar(cls, high: List[float], low: List[float], close: List[float],
                      af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.2) -> Tuple[float, bool]:
        n = min(len(high), len(low), len(close))
        if n < 5:
            return close[-1] if close else 0.0, True
        is_uptrend = close[-1] > close[0]
        af = af_start
        sar = min(low[:5]) if is_uptrend else max(high[:5])
        ep = max(high[:5]) if is_uptrend else min(low[:5])
        for i in range(5, n):
            if is_uptrend:
                sar = sar + af * (ep - sar)
                sar = min(sar, low[i-1], low[i-2])
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
                if low[i] < sar:
                    is_uptrend = False
                    sar = ep
                    ep = low[i]
                    af = af_start
            else:
                sar = sar + af * (ep - sar)
                sar = max(sar, high[i-1], high[i-2])
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)
                if high[i] > sar:
                    is_uptrend = True
                    sar = ep
                    ep = high[i]
                    af = af_start
        return sar, is_uptrend

    @classmethod
    def choppiness_index(cls, high: List[float], low: List[float], close: List[float], length: int = 14) -> float:
        n = min(len(high), len(low), len(close))
        if n < length + 1:
            return 50.0
        atr_sum = sum(cls.tr_series(high, low, close)[-length:])
        highest = max(high[-length:])
        lowest = min(low[-length:])
        if highest - lowest < cls.EPS:
            return 50.0
        chop = 100 * math.log10(atr_sum / (highest - lowest)) / math.log10(length)
        return max(0, min(100, chop))

    @classmethod
    def elder_ray(cls, high: List[float], low: List[float], close: List[float], length: int = 13) -> Tuple[float, float]:
        if len(close) < length:
            return 0.0, 0.0
        ema_val = cls.ema(close, length)
        bull_power = high[-1] - ema_val
        bear_power = low[-1] - ema_val
        return bull_power, bear_power

    @classmethod
    def klinger_oscillator(cls, high: List[float], low: List[float], close: List[float],
                           volume: List[float], fast: int = 34, slow: int = 55) -> Tuple[float, float]:
        n = min(len(high), len(low), len(close), len(volume))
        if n < slow + 1:
            return 0.0, 0.0
        trend = []
        dm = []
        cm = []
        vf = []
        for i in range(1, n):
            hlc_curr = high[i] + low[i] + close[i]
            hlc_prev = high[i-1] + low[i-1] + close[i-1]
            t = 1 if hlc_curr > hlc_prev else -1
            trend.append(t)
            dm_val = high[i] - low[i]
            dm.append(dm_val)
            if i == 1:
                cm.append(dm_val)
            else:
                if trend[-1] == trend[-2] if len(trend) >= 2 else True:
                    cm.append(cm[-1] + dm_val)
                else:
                    cm.append(dm_val)
            if abs(cm[-1]) < cls.EPS:
                vf.append(0.0)
            else:
                vf.append(volume[i] * abs(2 * (dm_val / cm[-1]) - 1) * t * 100)
        if len(vf) < slow:
            return 0.0, 0.0
        kvo = cls.ema(vf, fast) - cls.ema(vf, slow)
        signal = cls.ema(vf[-13:], 13) if len(vf) >= 13 else kvo
        return kvo, signal

    @classmethod
    def coppock_curve(cls, close: List[float], roc1: int = 14, roc2: int = 11, wma_length: int = 10) -> float:
        if len(close) < max(roc1, roc2) + wma_length:
            return 0.0
        roc_values = []
        for i in range(max(roc1, roc2), len(close)):
            r1 = ((close[i] - close[i - roc1]) / close[i - roc1]) * 100 if close[i - roc1] > cls.EPS else 0.0
            r2 = ((close[i] - close[i - roc2]) / close[i - roc2]) * 100 if close[i - roc2] > cls.EPS else 0.0
            roc_values.append(r1 + r2)
        return cls.wma(roc_values, wma_length)

    @classmethod
    def force_index(cls, close: List[float], volume: List[float], length: int = 13) -> float:
        n = min(len(close), len(volume))
        if n < 2:
            return 0.0
        fi_values = []
        for i in range(1, n):
            fi_values.append((close[i] - close[i-1]) * volume[i])
        return cls.ema(fi_values, length)

    @classmethod
    def ease_of_movement(cls, high: List[float], low: List[float], volume: List[float], length: int = 14) -> float:
        n = min(len(high), len(low), len(volume))
        if n < length + 1:
            return 0.0
        emv_values = []
        for i in range(1, n):
            dm = ((high[i] + low[i]) / 2) - ((high[i-1] + low[i-1]) / 2)
            br = (volume[i] / 10000) / ((high[i] - low[i]) + cls.EPS)
            emv_values.append(dm / (br + cls.EPS))
        return cls.sma(emv_values, length)

    @classmethod
    def mass_index(cls, high: List[float], low: List[float], ema_length: int = 9, sum_length: int = 25) -> float:
        n = min(len(high), len(low))
        if n < sum_length + ema_length:
            return 25.0
        hl_range = [high[i] - low[i] for i in range(n)]
        ema1 = [cls.ema(hl_range[:i+1], ema_length) for i in range(n)]
        ema2 = [cls.ema(ema1[:i+1], ema_length) for i in range(n)]
        ratio = [ema1[i] / (ema2[i] + cls.EPS) for i in range(n)]
        return sum(ratio[-sum_length:])

    @classmethod
    def vortex_indicator(cls, high: List[float], low: List[float], close: List[float], length: int = 14) -> Tuple[float, float]:
        n = min(len(high), len(low), len(close))
        if n < length + 1:
            return 1.0, 1.0
        vm_plus = []
        vm_minus = []
        tr_list = []
        for i in range(1, n):
            vm_plus.append(abs(high[i] - low[i-1]))
            vm_minus.append(abs(low[i] - high[i-1]))
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            tr_list.append(tr)
        sum_tr = sum(tr_list[-length:])
        if sum_tr < cls.EPS:
            return 1.0, 1.0
        vi_plus = sum(vm_plus[-length:]) / sum_tr
        vi_minus = sum(vm_minus[-length:]) / sum_tr
        return vi_plus, vi_minus

    @classmethod
    def donchian_channels(cls, high: List[float], low: List[float], length: int = 20) -> Tuple[float, float, float]:
        n = min(len(high), len(low))
        if n < length:
            if high and low:
                return high[-1], (high[-1] + low[-1]) / 2, low[-1]
            return 0.0, 0.0, 0.0
        upper = max(high[-length:])
        lower = min(low[-length:])
        middle = (upper + lower) / 2
        return upper, middle, lower

    @classmethod
    def supertrend(cls, high: List[float], low: List[float], close: List[float], 
                   length: int = 10, multiplier: float = 3.0) -> Tuple[float, bool]:
        n = min(len(high), len(low), len(close))
        if n < length + 1:
            return close[-1] if close else 0.0, True
        atr_val = cls.atr(high, low, close, length)
        hl2 = (high[-1] + low[-1]) / 2
        upper_band = hl2 + (multiplier * atr_val)
        lower_band = hl2 - (multiplier * atr_val)
        is_uptrend = close[-1] > lower_band
        supertrend = lower_band if is_uptrend else upper_band
        return supertrend, is_uptrend

    @classmethod
    def accumulation_distribution(cls, high: List[float], low: List[float], close: List[float], volume: List[float]) -> float:
        n = min(len(high), len(low), len(close), len(volume))
        if n < 1:
            return 0.0
        ad = 0.0
        for i in range(n):
            hl_range = high[i] - low[i]
            if hl_range > cls.EPS:
                clv = ((close[i] - low[i]) - (high[i] - close[i])) / hl_range
                ad += clv * volume[i]
        return ad

    @classmethod
    def linear_regression(cls, close: List[float], length: int = 14) -> Tuple[float, float, float]:
        if len(close) < length:
            return close[-1] if close else 0.0, 0.0, 0.0
        y = close[-length:]
        x = list(range(length))
        mx = (length - 1) / 2
        my = sum(y) / length
        num = sum((x[i] - mx) * (y[i] - my) for i in range(length))
        den = sum((xi - mx) ** 2 for xi in x) or 1.0
        slope = num / den
        intercept = my - slope * mx
        value = intercept + slope * (length - 1)
        r_squared = 0.0
        ss_tot = sum((yi - my) ** 2 for yi in y)
        if ss_tot > cls.EPS:
            ss_res = sum((y[i] - (intercept + slope * x[i])) ** 2 for i in range(length))
            r_squared = 1 - (ss_res / ss_tot)
        return value, slope, r_squared

    @classmethod
    def standard_deviation(cls, close: List[float], length: int = 20) -> float:
        if len(close) < length:
            return 0.0
        recent = close[-length:]
        mean = sum(recent) / length
        variance = sum((x - mean) ** 2 for x in recent) / length
        return math.sqrt(variance)

    @classmethod
    def coefficient_of_variation(cls, close: List[float], length: int = 20) -> float:
        if len(close) < length:
            return 0.0
        mean = sum(close[-length:]) / length
        if abs(mean) < cls.EPS:
            return 0.0
        std = cls.standard_deviation(close, length)
        return (std / abs(mean)) * 100

    @classmethod
    def schaff_trend_cycle(cls, close: List[float], fast: int = 23, slow: int = 50, cycle: int = 10) -> float:
        if len(close) < slow + cycle:
            return 50.0
        ema_fast = cls.ema(close, fast)
        ema_slow = cls.ema(close, slow)
        macd_val = ema_fast - ema_slow
        macd_series = [0.0] * len(close)
        for i in range(slow, len(close)):
            ef = cls.ema(close[:i+1], fast)
            es = cls.ema(close[:i+1], slow)
            macd_series[i] = ef - es
        recent = macd_series[-cycle:]
        low_macd = min(recent) if recent else 0
        high_macd = max(recent) if recent else 0
        if high_macd - low_macd == 0:
            return 50.0
        stoch1 = ((macd_val - low_macd) / (high_macd - low_macd)) * 100
        return max(0, min(100, stoch1))

    @classmethod
    def fisher_transform(cls, close: List[float], length: int = 9) -> float:
        if len(close) < length:
            return 0.0
        high_val = max(close[-length:])
        low_val = min(close[-length:])
        if high_val == low_val:
            return 0.0
        value = 2 * ((close[-1] - low_val) / (high_val - low_val)) - 1
        value = max(-0.999, min(0.999, value * 0.67))
        fisher = 0.5 * math.log((1 + value) / (1 - value))
        return round(fisher, 4)

    @classmethod
    def detrended_price_oscillator(cls, close: List[float], length: int = 20) -> float:
        if len(close) < length + length // 2 + 1:
            return 0.0
        sma = cls.sma(close[:-length//2-1], length)
        return round(close[-length//2-1] - sma, 6)

    @classmethod
    def know_sure_thing(cls, close: List[float]) -> Tuple[float, float]:
        if len(close) < 30:
            return 0.0, 0.0
        roc1 = cls.rate_of_change(close[-10:], 10)
        roc2 = cls.rate_of_change(close[-15:], 15)
        roc3 = cls.rate_of_change(close[-20:], 20)
        roc4 = cls.rate_of_change(close[-30:], 30)
        kst = roc1 * 1 + roc2 * 2 + roc3 * 3 + roc4 * 4
        signal = cls.ema([kst] * 9, 9)
        return round(kst, 4), round(signal, 4)

    @classmethod
    def chande_momentum(cls, close: List[float], length: int = 9) -> float:
        if len(close) < length + 1:
            return 0.0
        gains = []
        losses = []
        for i in range(-length, 0):
            change = close[i] - close[i-1]
            if change > 0:
                gains.append(change)
            else:
                losses.append(abs(change))
        sum_gains = sum(gains)
        sum_losses = sum(losses)
        if sum_gains + sum_losses == 0:
            return 0.0
        return round(((sum_gains - sum_losses) / (sum_gains + sum_losses)) * 100, 2)

    @classmethod
    def relative_vigor_index(cls, high: List[float], low: List[float], close: List[float], 
                              open_prices: Optional[List[float]] = None, length: int = 10) -> float:
        if len(close) < length + 3:
            return 0.0
        if open_prices is None:
            open_prices = [close[max(0,i-1)] for i in range(len(close))]
        numerator = 0.0
        denominator = 0.0
        for i in range(-length, 0):
            co = close[i] - open_prices[i]
            hl = high[i] - low[i]
            numerator += co
            denominator += hl
        if denominator == 0:
            return 0.0
        return round(numerator / denominator, 4)

    @classmethod
    def percentage_price_oscillator(cls, close: List[float], fast: int = 12, slow: int = 26) -> Tuple[float, float, float]:
        if len(close) < slow:
            return 0.0, 0.0, 0.0
        ema_fast = cls.ema(close, fast)
        ema_slow = cls.ema(close, slow)
        if ema_slow == 0:
            return 0.0, 0.0, 0.0
        ppo = ((ema_fast - ema_slow) / ema_slow) * 100
        signal = cls.ema([ppo] * 9, 9)
        histogram = ppo - signal
        return round(ppo, 4), round(signal, 4), round(histogram, 4)

    @classmethod
    def balance_of_power(cls, high: List[float], low: List[float], close: List[float],
                         open_prices: Optional[List[float]] = None) -> float:
        if not close or len(close) < 1:
            return 0.0
        if open_prices is None:
            open_prices = [close[max(0,i-1)] for i in range(len(close))]
        hl_diff = high[-1] - low[-1]
        if hl_diff == 0:
            return 0.0
        return round((close[-1] - open_prices[-1]) / hl_diff, 4)

    @classmethod
    def price_momentum_oscillator(cls, close: List[float], length1: int = 35, length2: int = 20) -> float:
        if len(close) < length1 + length2:
            return 0.0
        roc = cls.rate_of_change(close, 1)
        smoothed1 = cls.ema([roc] * length1, length1)
        smoothed2 = cls.ema([smoothed1] * length2, length2)
        return round(smoothed2 * 10, 4)

    @classmethod
    def calculate_all(cls, high: List[float], low: List[float], close: List[float], 
                      volume: List[float]) -> dict:
        aroon_up, aroon_down, aroon_osc = cls.aroon(high, low)
        vi_plus, vi_minus = cls.vortex_indicator(high, low, close)
        supertrend_val, supertrend_up = cls.supertrend(high, low, close)
        donchian_upper, donchian_mid, donchian_lower = cls.donchian_channels(high, low)
        sar_val, sar_uptrend = cls.parabolic_sar(high, low, close)
        bull_power, bear_power = cls.elder_ray(high, low, close)
        kvo, kvo_signal = cls.klinger_oscillator(high, low, close, volume)
        linreg_val, linreg_slope, linreg_r2 = cls.linear_regression(close)
        kst_val, kst_signal = cls.know_sure_thing(close)
        ppo_val, ppo_signal, ppo_hist = cls.percentage_price_oscillator(close)
        
        return {
            "rsi": cls.rsi(close),
            "stoch_rsi": cls.stochastic_rsi(close),
            "macd": cls.macd(close),
            "bollinger": cls.bollinger_bands(close),
            "keltner": cls.keltner_channels(high, low, close),
            "ttm_squeeze": cls.ttm_squeeze(high, low, close),
            "adx": cls.adx(high, low, close),
            "obv": cls.obv(close, volume),
            "vwap": cls.vwap(high, low, close, volume),
            "mfi": cls.mfi(high, low, close, volume),
            "cci": cls.cci(high, low, close),
            "williams_r": cls.williams_r(high, low, close),
            "ichimoku": cls.ichimoku(high, low, close),
            "atr": cls.atr(high, low, close),
            "breakout_energy": cls.breakout_energy(high, low, close, volume),
            "momentum": cls.momentum_oscillator(close),
            "roc": cls.rate_of_change(close),
            "cmf": cls.chaikin_money_flow(high, low, close, volume),
            "volume_profile": cls.volume_profile(close, volume),
            "dema": cls.dema(close, 20),
            "tema": cls.tema(close, 20),
            "trix": cls.trix(close),
            "ultimate_oscillator": cls.ultimate_oscillator(high, low, close),
            "aroon_up": aroon_up,
            "aroon_down": aroon_down,
            "aroon_osc": aroon_osc,
            "choppiness": cls.choppiness_index(high, low, close),
            "bull_power": bull_power,
            "bear_power": bear_power,
            "coppock": cls.coppock_curve(close),
            "force_index": cls.force_index(close, volume),
            "ease_of_movement": cls.ease_of_movement(high, low, volume),
            "mass_index": cls.mass_index(high, low),
            "vortex_plus": vi_plus,
            "vortex_minus": vi_minus,
            "donchian_upper": donchian_upper,
            "donchian_mid": donchian_mid,
            "donchian_lower": donchian_lower,
            "supertrend": supertrend_val,
            "supertrend_bullish": supertrend_up,
            "parabolic_sar": sar_val,
            "sar_uptrend": sar_uptrend,
            "acc_dist": cls.accumulation_distribution(high, low, close, volume),
            "linreg_value": linreg_val,
            "linreg_slope": linreg_slope,
            "linreg_r2": linreg_r2,
            "std_dev": cls.standard_deviation(close),
            "coef_variation": cls.coefficient_of_variation(close),
            "klinger": kvo,
            "klinger_signal": kvo_signal,
            "hull_ma": cls.hull_ma(close, 20),
            "schaff_trend_cycle": cls.schaff_trend_cycle(close),
            "fisher_transform": cls.fisher_transform(close),
            "dpo": cls.detrended_price_oscillator(close),
            "kst": kst_val,
            "kst_signal": kst_signal,
            "chande_momentum": cls.chande_momentum(close),
            "rvi": cls.relative_vigor_index(high, low, close),
            "ppo": ppo_val,
            "ppo_signal": ppo_signal,
            "ppo_histogram": ppo_hist,
            "balance_of_power": cls.balance_of_power(high, low, close),
            "pmo": cls.price_momentum_oscillator(close)
        }
