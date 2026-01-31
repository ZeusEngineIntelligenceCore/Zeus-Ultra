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
        return ((close[-1] / close[-length - 1]) - 1) * 100

    @classmethod
    def rate_of_change(cls, close: List[float], length: int = 12) -> float:
        if len(close) < length + 1:
            return 0.0
        return ((close[-1] - close[-length - 1]) / close[-length - 1]) * 100

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
    def calculate_all(cls, high: List[float], low: List[float], close: List[float], 
                      volume: List[float]) -> dict:
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
            "volume_profile": cls.volume_profile(close, volume)
        }
