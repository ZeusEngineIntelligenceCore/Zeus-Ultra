#!/usr/bin/env python3
"""
ZEUS ADVANCED MATHEMATICAL ANALYSIS
Cutting-edge quantitative indicators:
- Hurst Exponent (trend persistence)
- Fractal Support/Resistance detection
- Wavelet Denoising for noise filtering
- Entropy-based market efficiency
- Information Ratio analysis
"""

from __future__ import annotations
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np


@dataclass
class FractalLevel:
    price: float
    level_type: str
    strength: float
    touch_count: int
    last_touch_index: int


@dataclass
class AdvancedMathSignals:
    hurst_exponent: float
    trend_persistence: str
    fractal_supports: List[FractalLevel]
    fractal_resistances: List[FractalLevel]
    nearest_support: float
    nearest_resistance: float
    denoised_trend: str
    denoised_price: float
    market_entropy: float
    market_efficiency: str
    information_ratio: float
    signals: List[str] = field(default_factory=list)
    overall_score: float = 50.0


class HurstExponentCalculator:
    @staticmethod
    def calculate(prices: List[float], max_lag: int = 100) -> float:
        if len(prices) < 20:
            return 0.5
        
        n = len(prices)
        max_lag = min(max_lag, n // 4)
        
        if max_lag < 4:
            return 0.5
        
        lags = list(range(4, max_lag + 1))
        rs_values = []
        
        for lag in lags:
            rs_sum = 0.0
            count = 0
            
            for start in range(0, n - lag, lag):
                segment = prices[start:start + lag]
                
                if len(segment) < lag:
                    continue
                
                mean = sum(segment) / len(segment)
                deviations = [p - mean for p in segment]
                
                cumulative = []
                cumsum = 0
                for d in deviations:
                    cumsum += d
                    cumulative.append(cumsum)
                
                R = max(cumulative) - min(cumulative)
                
                S = math.sqrt(sum(d ** 2 for d in deviations) / len(deviations))
                
                if S > 0:
                    rs_sum += R / S
                    count += 1
            
            if count > 0:
                rs_values.append(rs_sum / count)
            else:
                rs_values.append(1.0)
        
        if len(lags) < 2 or len(rs_values) < 2:
            return 0.5
        
        log_lags = [math.log(lag) for lag in lags]
        log_rs = [math.log(max(rs, 1e-10)) for rs in rs_values]
        
        n_points = len(log_lags)
        mean_x = sum(log_lags) / n_points
        mean_y = sum(log_rs) / n_points
        
        numerator = sum((log_lags[i] - mean_x) * (log_rs[i] - mean_y) for i in range(n_points))
        denominator = sum((log_lags[i] - mean_x) ** 2 for i in range(n_points))
        
        if denominator == 0:
            return 0.5
        
        hurst = numerator / denominator
        
        return max(0.0, min(1.0, hurst))
    
    @staticmethod
    def interpret(hurst: float) -> str:
        if hurst > 0.6:
            return "trending"
        elif hurst < 0.4:
            return "mean_reverting"
        else:
            return "random_walk"


class FractalDetector:
    def __init__(self, lookback: int = 5, min_touches: int = 2, tolerance_pct: float = 0.005):
        self.lookback = lookback
        self.min_touches = min_touches
        self.tolerance_pct = tolerance_pct
    
    def find_fractals(self, highs: List[float], lows: List[float]) -> Tuple[List[int], List[int]]:
        if len(highs) < self.lookback * 2 + 1:
            return [], []
        
        fractal_highs = []
        fractal_lows = []
        
        for i in range(self.lookback, len(highs) - self.lookback):
            is_fractal_high = True
            for j in range(1, self.lookback + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_fractal_high = False
                    break
            
            if is_fractal_high:
                fractal_highs.append(i)
        
        for i in range(self.lookback, len(lows) - self.lookback):
            is_fractal_low = True
            for j in range(1, self.lookback + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_fractal_low = False
                    break
            
            if is_fractal_low:
                fractal_lows.append(i)
        
        return fractal_highs, fractal_lows
    
    def cluster_levels(self, prices: List[float], indices: List[int], 
                      all_prices: List[float]) -> List[FractalLevel]:
        if not indices or not prices:
            return []
        
        fractal_prices = [(prices[i], i) for i in indices if i < len(prices)]
        
        if not fractal_prices:
            return []
        
        fractal_prices.sort(key=lambda x: x[0])
        
        clusters = []
        current_cluster = [fractal_prices[0]]
        
        for i in range(1, len(fractal_prices)):
            price, idx = fractal_prices[i]
            cluster_avg = sum(p for p, _ in current_cluster) / len(current_cluster)
            
            if abs(price - cluster_avg) / cluster_avg <= self.tolerance_pct:
                current_cluster.append((price, idx))
            else:
                clusters.append(current_cluster)
                current_cluster = [(price, idx)]
        
        clusters.append(current_cluster)
        
        levels = []
        for cluster in clusters:
            if len(cluster) >= self.min_touches:
                avg_price = sum(p for p, _ in cluster) / len(cluster)
                max_idx = max(idx for _, idx in cluster)
                
                touch_count = self._count_touches(all_prices, avg_price)
                
                strength = min(1.0, len(cluster) / 5) * min(1.0, touch_count / 10)
                
                levels.append(FractalLevel(
                    price=avg_price,
                    level_type="resistance" if avg_price > all_prices[-1] else "support",
                    strength=strength,
                    touch_count=touch_count,
                    last_touch_index=max_idx
                ))
        
        return levels
    
    def _count_touches(self, prices: List[float], level: float) -> int:
        touches = 0
        in_zone = False
        
        for price in prices:
            price_in_zone = abs(price - level) / level <= self.tolerance_pct
            
            if price_in_zone and not in_zone:
                touches += 1
                in_zone = True
            elif not price_in_zone:
                in_zone = False
        
        return touches
    
    def find_support_resistance(self, highs: List[float], lows: List[float], 
                               closes: List[float]) -> Tuple[List[FractalLevel], List[FractalLevel]]:
        fractal_high_indices, fractal_low_indices = self.find_fractals(highs, lows)
        
        resistances = self.cluster_levels(highs, fractal_high_indices, closes)
        supports = self.cluster_levels(lows, fractal_low_indices, closes)
        
        current_price = closes[-1] if closes else 0
        
        resistances = [r for r in resistances if r.price > current_price]
        supports = [s for s in supports if s.price < current_price]
        
        resistances.sort(key=lambda x: x.price)
        supports.sort(key=lambda x: -x.price)
        
        return supports[:5], resistances[:5]


class WaveletDenoiser:
    @staticmethod
    def haar_decompose(signal: List[float], levels: int = 3) -> Tuple[List[float], List[List[float]]]:
        if len(signal) < 2 ** levels:
            return signal, []
        
        approximation = signal.copy()
        details = []
        
        for level in range(levels):
            n = len(approximation)
            if n < 2:
                break
            
            new_n = n // 2
            new_approx = []
            detail = []
            
            for i in range(new_n):
                avg = (approximation[2*i] + approximation[2*i + 1]) / 2
                diff = (approximation[2*i] - approximation[2*i + 1]) / 2
                new_approx.append(avg)
                detail.append(diff)
            
            approximation = new_approx
            details.append(detail)
        
        return approximation, details
    
    @staticmethod
    def haar_reconstruct(approximation: List[float], details: List[List[float]]) -> List[float]:
        signal = approximation.copy()
        
        for detail in reversed(details):
            new_signal = []
            for i in range(len(signal)):
                d = detail[i] if i < len(detail) else 0
                new_signal.append(signal[i] + d)
                new_signal.append(signal[i] - d)
            signal = new_signal
        
        return signal
    
    @staticmethod
    def soft_threshold(details: List[List[float]], threshold_mult: float = 1.0) -> List[List[float]]:
        thresholded = []
        
        for detail in details:
            if not detail:
                thresholded.append([])
                continue
            
            mad = sorted([abs(d) for d in detail])[len(detail) // 2]
            threshold = threshold_mult * mad * 1.4826
            
            new_detail = []
            for d in detail:
                if abs(d) <= threshold:
                    new_detail.append(0)
                elif d > 0:
                    new_detail.append(d - threshold)
                else:
                    new_detail.append(d + threshold)
            
            thresholded.append(new_detail)
        
        return thresholded
    
    @classmethod
    def denoise(cls, prices: List[float], levels: int = 3, 
                threshold_mult: float = 1.0) -> List[float]:
        if len(prices) < 8:
            return prices
        
        power_of_2 = 2 ** int(math.log2(len(prices)))
        padded = prices[-power_of_2:]
        
        approx, details = cls.haar_decompose(padded, levels)
        
        thresholded_details = cls.soft_threshold(details, threshold_mult)
        
        denoised = cls.haar_reconstruct(approx, thresholded_details)
        
        return denoised


class EntropyCalculator:
    @staticmethod
    def shannon_entropy(prices: List[float], bins: int = 20) -> float:
        if len(prices) < 10:
            return 0.5
        
        returns = [(prices[i] - prices[i-1]) / prices[i-1] 
                   for i in range(1, len(prices)) if prices[i-1] != 0]
        
        if not returns:
            return 0.5
        
        min_ret = min(returns)
        max_ret = max(returns)
        
        if max_ret == min_ret:
            return 0.0
        
        bin_size = (max_ret - min_ret) / bins
        histogram = [0] * bins
        
        for ret in returns:
            bin_idx = min(int((ret - min_ret) / bin_size), bins - 1)
            histogram[bin_idx] += 1
        
        total = len(returns)
        entropy = 0.0
        
        for count in histogram:
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        max_entropy = math.log2(bins)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.5
        
        return normalized_entropy
    
    @staticmethod
    def interpret_entropy(entropy: float) -> str:
        if entropy > 0.8:
            return "highly_random"
        elif entropy > 0.6:
            return "moderately_efficient"
        elif entropy > 0.4:
            return "predictable_patterns"
        else:
            return "highly_structured"


class InformationRatioCalculator:
    @staticmethod
    def calculate(returns: List[float], benchmark_returns: Optional[List[float]] = None) -> float:
        if len(returns) < 10:
            return 0.0
        
        if benchmark_returns is None:
            benchmark_returns = [0.0] * len(returns)
        
        min_len = min(len(returns), len(benchmark_returns))
        active_returns = [returns[i] - benchmark_returns[i] for i in range(min_len)]
        
        mean_active = sum(active_returns) / len(active_returns)
        
        tracking_error = math.sqrt(
            sum((r - mean_active) ** 2 for r in active_returns) / len(active_returns)
        )
        
        if tracking_error == 0:
            return 0.0
        
        annualized_mean = mean_active * 365
        annualized_te = tracking_error * math.sqrt(365)
        
        return annualized_mean / annualized_te


class AdvancedMathAnalyzer:
    def __init__(self):
        self.hurst_calc = HurstExponentCalculator()
        self.fractal_detector = FractalDetector()
        self.wavelet_denoiser = WaveletDenoiser()
        self.entropy_calc = EntropyCalculator()
        self.ir_calc = InformationRatioCalculator()
    
    def analyze(self, highs: List[float], lows: List[float], 
                closes: List[float], volumes: List[float]) -> AdvancedMathSignals:
        signals = []
        overall_score = 50.0
        
        hurst = self.hurst_calc.calculate(closes)
        trend_persistence = self.hurst_calc.interpret(hurst)
        
        if trend_persistence == "trending":
            signals.append(f"Hurst={hurst:.2f}: Strong trend persistence - follow the trend")
            overall_score += 10
        elif trend_persistence == "mean_reverting":
            signals.append(f"Hurst={hurst:.2f}: Mean-reverting market - fade extremes")
            overall_score -= 5
        
        supports, resistances = self.fractal_detector.find_support_resistance(highs, lows, closes)
        
        current_price = closes[-1] if closes else 0
        nearest_support = supports[0].price if supports else current_price * 0.95
        nearest_resistance = resistances[0].price if resistances else current_price * 1.05
        
        distance_to_support = (current_price - nearest_support) / current_price if current_price > 0 else 0
        distance_to_resistance = (nearest_resistance - current_price) / current_price if current_price > 0 else 0
        
        if distance_to_support < 0.01:
            signals.append(f"Near strong support at ${nearest_support:.6f}")
            overall_score += 15
        if distance_to_resistance < 0.01:
            signals.append(f"Near strong resistance at ${nearest_resistance:.6f}")
            overall_score -= 10
        
        if supports:
            strongest_support = max(supports, key=lambda x: x.strength)
            if strongest_support.strength > 0.7:
                signals.append(f"Strong support zone: ${strongest_support.price:.6f} ({strongest_support.touch_count} touches)")
        
        denoised = self.wavelet_denoiser.denoise(closes)
        if len(denoised) >= 2:
            denoised_price = denoised[-1]
            denoised_trend = "bullish" if denoised[-1] > denoised[-2] else "bearish"
            
            if denoised_trend == "bullish":
                signals.append("Denoised trend is bullish")
                overall_score += 5
        else:
            denoised_price = current_price
            denoised_trend = "neutral"
        
        entropy = self.entropy_calc.shannon_entropy(closes)
        market_efficiency = self.entropy_calc.interpret_entropy(entropy)
        
        if market_efficiency == "highly_structured":
            signals.append("Market showing predictable patterns")
            overall_score += 10
        elif market_efficiency == "highly_random":
            signals.append("Market highly random - reduce position sizes")
            overall_score -= 10
        
        returns = [(closes[i] - closes[i-1]) / closes[i-1] 
                   for i in range(1, len(closes)) if closes[i-1] != 0]
        information_ratio = self.ir_calc.calculate(returns)
        
        if information_ratio > 1.0:
            signals.append(f"High information ratio: {information_ratio:.2f}")
            overall_score += 5
        
        overall_score = max(0, min(100, overall_score))
        
        return AdvancedMathSignals(
            hurst_exponent=hurst,
            trend_persistence=trend_persistence,
            fractal_supports=supports,
            fractal_resistances=resistances,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            denoised_trend=denoised_trend,
            denoised_price=denoised_price,
            market_entropy=entropy,
            market_efficiency=market_efficiency,
            information_ratio=information_ratio,
            signals=signals,
            overall_score=overall_score
        )
