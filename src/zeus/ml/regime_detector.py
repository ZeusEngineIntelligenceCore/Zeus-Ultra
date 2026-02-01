#!/usr/bin/env python3
"""
ZEUS MARKET REGIME DETECTOR
Advanced regime detection using:
- Hidden Markov Models (HMM) for state transitions
- GARCH volatility clustering
- Trend vs Range classification
- Momentum vs Mean-Reversion mode switching
"""

from __future__ import annotations
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class MarketRegime(Enum):
    STRONG_TREND_UP = "strong_trend_up"
    WEAK_TREND_UP = "weak_trend_up"
    RANGING = "ranging"
    WEAK_TREND_DOWN = "weak_trend_down"
    STRONG_TREND_DOWN = "strong_trend_down"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"


class TradingMode(Enum):
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    DEFENSIVE = "defensive"
    AGGRESSIVE = "aggressive"


@dataclass
class RegimeConfig:
    lookback_short: int = 20
    lookback_medium: int = 50
    lookback_long: int = 200
    volatility_window: int = 20
    trend_threshold: float = 0.02
    strong_trend_threshold: float = 0.05
    volatility_high_threshold: float = 0.03
    volatility_low_threshold: float = 0.01
    ranging_threshold: float = 0.015
    hmm_states: int = 5
    garch_omega: float = 0.00001
    garch_alpha: float = 0.1
    garch_beta: float = 0.85


@dataclass
class RegimeState:
    regime: MarketRegime
    confidence: float
    trading_mode: TradingMode
    volatility_state: str
    trend_strength: float
    momentum_score: float
    mean_reversion_score: float
    regime_duration: int
    transition_probability: float
    recommended_position_size_mult: float
    recommended_stop_mult: float
    signals: List[str] = field(default_factory=list)


class HiddenMarkovModel:
    def __init__(self, n_states: int = 5):
        self.n_states = n_states
        self.transition_matrix = self._init_transition_matrix()
        self.emission_means = np.linspace(-0.03, 0.03, n_states)
        self.emission_stds = np.array([0.02] * n_states)
        self.state_probs = np.array([1.0 / n_states] * n_states)
        self.current_state = n_states // 2
        
    def _init_transition_matrix(self) -> np.ndarray:
        matrix = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            matrix[i, i] = 0.7
            if i > 0:
                matrix[i, i-1] = 0.15
            if i < self.n_states - 1:
                matrix[i, i+1] = 0.15
            row_sum = matrix[i].sum()
            if row_sum > 0:
                matrix[i] /= row_sum
        return matrix
    
    def _gaussian_pdf(self, x: float, mean: float, std: float) -> float:
        if std <= 0:
            return 0.0
        coefficient = 1.0 / (std * math.sqrt(2 * math.pi))
        exponent = -0.5 * ((x - mean) / std) ** 2
        return coefficient * math.exp(exponent)
    
    def update(self, observation: float) -> int:
        emission_probs = np.array([
            self._gaussian_pdf(observation, self.emission_means[i], self.emission_stds[i])
            for i in range(self.n_states)
        ])
        
        emission_probs = np.clip(emission_probs, 1e-10, None)
        
        new_probs = np.zeros(self.n_states)
        for j in range(self.n_states):
            for i in range(self.n_states):
                new_probs[j] += self.state_probs[i] * self.transition_matrix[i, j]
            new_probs[j] *= emission_probs[j]
        
        prob_sum = new_probs.sum()
        if prob_sum > 0:
            new_probs /= prob_sum
        else:
            new_probs = np.array([1.0 / self.n_states] * self.n_states)
        
        self.state_probs = new_probs
        self.current_state = int(np.argmax(self.state_probs))
        
        return self.current_state
    
    def get_transition_probability(self, from_state: int, to_state: int) -> float:
        if 0 <= from_state < self.n_states and 0 <= to_state < self.n_states:
            return float(self.transition_matrix[from_state, to_state])
        return 0.0
    
    def get_state_confidence(self) -> float:
        return float(self.state_probs[self.current_state])
    
    def adapt_emissions(self, returns: List[float]) -> None:
        if len(returns) < 50:
            return
        
        sorted_returns = sorted(returns)
        n = len(sorted_returns)
        
        for i in range(self.n_states):
            idx = int((i + 0.5) / self.n_states * n)
            idx = min(idx, n - 1)
            self.emission_means[i] = sorted_returns[idx]
        
        overall_std = np.std(returns)
        self.emission_stds = np.array([overall_std] * self.n_states)


class GARCHModel:
    def __init__(self, omega: float = 0.00001, alpha: float = 0.1, beta: float = 0.85):
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.current_variance = omega / (1 - alpha - beta) if (alpha + beta) < 1 else omega
        self.variance_history: List[float] = []
        
    def update(self, return_val: float) -> float:
        new_variance = self.omega + self.alpha * (return_val ** 2) + self.beta * self.current_variance
        
        new_variance = max(new_variance, 1e-10)
        
        self.current_variance = new_variance
        self.variance_history.append(new_variance)
        
        if len(self.variance_history) > 500:
            self.variance_history = self.variance_history[-500:]
        
        return math.sqrt(new_variance)
    
    def get_volatility(self) -> float:
        return math.sqrt(self.current_variance)
    
    def forecast_volatility(self, steps: int = 5) -> List[float]:
        forecasts = []
        variance = self.current_variance
        
        long_term_var = self.omega / (1 - self.alpha - self.beta) if (self.alpha + self.beta) < 1 else self.omega
        
        for _ in range(steps):
            variance = self.omega + (self.alpha + self.beta) * variance
            variance = 0.9 * variance + 0.1 * long_term_var
            forecasts.append(math.sqrt(variance))
        
        return forecasts
    
    def is_volatility_clustering(self) -> bool:
        if len(self.variance_history) < 20:
            return False
        
        recent = self.variance_history[-10:]
        older = self.variance_history[-20:-10]
        
        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        
        return recent_avg > older_avg * 1.5 or recent_avg < older_avg * 0.67


class TrendClassifier:
    def __init__(self, config: RegimeConfig):
        self.config = config
        
    def classify(self, prices: List[float]) -> Tuple[str, float]:
        if len(prices) < self.config.lookback_long:
            return "unknown", 0.0
        
        short_ma = sum(prices[-self.config.lookback_short:]) / self.config.lookback_short
        medium_ma = sum(prices[-self.config.lookback_medium:]) / self.config.lookback_medium
        long_ma = sum(prices[-self.config.lookback_long:]) / self.config.lookback_long
        
        current = prices[-1]
        
        short_slope = (prices[-1] - prices[-self.config.lookback_short]) / prices[-self.config.lookback_short] if len(prices) >= self.config.lookback_short else 0
        medium_slope = (prices[-1] - prices[-self.config.lookback_medium]) / prices[-self.config.lookback_medium] if len(prices) >= self.config.lookback_medium else 0
        
        ma_alignment = 0
        if short_ma > medium_ma > long_ma:
            ma_alignment = 1
        elif short_ma < medium_ma < long_ma:
            ma_alignment = -1
        
        if ma_alignment == 1 and short_slope > self.config.strong_trend_threshold:
            return "strong_uptrend", min(1.0, short_slope / 0.1)
        elif ma_alignment == 1 and short_slope > self.config.trend_threshold:
            return "uptrend", short_slope / self.config.strong_trend_threshold
        elif ma_alignment == -1 and short_slope < -self.config.strong_trend_threshold:
            return "strong_downtrend", min(1.0, abs(short_slope) / 0.1)
        elif ma_alignment == -1 and short_slope < -self.config.trend_threshold:
            return "downtrend", abs(short_slope) / self.config.strong_trend_threshold
        else:
            return "ranging", 1.0 - abs(short_slope) / self.config.trend_threshold


class MeanReversionDetector:
    def __init__(self, config: RegimeConfig):
        self.config = config
        
    def calculate_score(self, prices: List[float]) -> float:
        if len(prices) < self.config.lookback_medium:
            return 0.5
        
        mean_price = sum(prices[-self.config.lookback_medium:]) / self.config.lookback_medium
        current = prices[-1]
        
        std_price = math.sqrt(sum((p - mean_price) ** 2 for p in prices[-self.config.lookback_medium:]) / self.config.lookback_medium)
        
        if std_price == 0:
            return 0.5
        
        z_score = (current - mean_price) / std_price
        
        mr_score = 1.0 / (1.0 + math.exp(-abs(z_score) + 2))
        
        if abs(z_score) > 2:
            mr_score = min(1.0, mr_score * 1.5)
        
        return mr_score
    
    def get_mean_reversion_target(self, prices: List[float]) -> float:
        if len(prices) < self.config.lookback_medium:
            return prices[-1] if prices else 0
        
        return sum(prices[-self.config.lookback_medium:]) / self.config.lookback_medium


class RegimeDetector:
    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self.hmm = HiddenMarkovModel(self.config.hmm_states)
        self.garch = GARCHModel(
            self.config.garch_omega,
            self.config.garch_alpha,
            self.config.garch_beta
        )
        self.trend_classifier = TrendClassifier(self.config)
        self.mr_detector = MeanReversionDetector(self.config)
        self.price_history: List[float] = []
        self.return_history: List[float] = []
        self.regime_history: List[MarketRegime] = []
        self.current_regime_duration = 0
        
    def update(self, price: float) -> None:
        if self.price_history:
            ret = (price - self.price_history[-1]) / self.price_history[-1] if self.price_history[-1] > 0 else 0
            self.return_history.append(ret)
            
            self.hmm.update(ret)
            self.garch.update(ret)
            
            if len(self.return_history) > 500:
                self.return_history = self.return_history[-500:]
        
        self.price_history.append(price)
        if len(self.price_history) > 500:
            self.price_history = self.price_history[-500:]
    
    def detect_regime(self) -> RegimeState:
        if len(self.price_history) < 50:
            return RegimeState(
                regime=MarketRegime.RANGING,
                confidence=0.5,
                trading_mode=TradingMode.DEFENSIVE,
                volatility_state="unknown",
                trend_strength=0.0,
                momentum_score=0.5,
                mean_reversion_score=0.5,
                regime_duration=0,
                transition_probability=0.5,
                recommended_position_size_mult=0.5,
                recommended_stop_mult=1.5,
                signals=["Insufficient data for regime detection"]
            )
        
        trend_type, trend_strength = self.trend_classifier.classify(self.price_history)
        
        current_vol = self.garch.get_volatility()
        annualized_vol = current_vol * math.sqrt(365)
        
        if annualized_vol > self.config.volatility_high_threshold * 20:
            volatility_state = "high"
        elif annualized_vol < self.config.volatility_low_threshold * 20:
            volatility_state = "low"
        else:
            volatility_state = "normal"
        
        hmm_state = self.hmm.current_state
        hmm_confidence = self.hmm.get_state_confidence()
        
        mr_score = self.mr_detector.calculate_score(self.price_history)
        
        if len(self.return_history) >= 10:
            recent_returns = self.return_history[-10:]
            momentum_score = (sum(1 for r in recent_returns if r > 0) / len(recent_returns))
        else:
            momentum_score = 0.5
        
        if trend_type == "strong_uptrend":
            regime = MarketRegime.STRONG_TREND_UP
        elif trend_type == "uptrend":
            regime = MarketRegime.WEAK_TREND_UP
        elif trend_type == "strong_downtrend":
            regime = MarketRegime.STRONG_TREND_DOWN
        elif trend_type == "downtrend":
            regime = MarketRegime.WEAK_TREND_DOWN
        elif volatility_state == "high":
            regime = MarketRegime.HIGH_VOLATILITY
        elif volatility_state == "low":
            regime = MarketRegime.LOW_VOLATILITY
        else:
            regime = MarketRegime.RANGING
        
        if regime in [MarketRegime.STRONG_TREND_UP, MarketRegime.STRONG_TREND_DOWN]:
            trading_mode = TradingMode.MOMENTUM
        elif regime == MarketRegime.RANGING and mr_score > 0.7:
            trading_mode = TradingMode.MEAN_REVERSION
        elif volatility_state == "low" and regime == MarketRegime.RANGING:
            trading_mode = TradingMode.BREAKOUT
        elif volatility_state == "high":
            trading_mode = TradingMode.DEFENSIVE
        else:
            trading_mode = TradingMode.MOMENTUM if momentum_score > 0.6 else TradingMode.MEAN_REVERSION
        
        if self.regime_history and self.regime_history[-1] == regime:
            self.current_regime_duration += 1
        else:
            self.current_regime_duration = 1
        
        self.regime_history.append(regime)
        if len(self.regime_history) > 100:
            self.regime_history = self.regime_history[-100:]
        
        if regime in [MarketRegime.STRONG_TREND_UP, MarketRegime.STRONG_TREND_DOWN]:
            position_mult = 1.2
            stop_mult = 0.8
        elif regime == MarketRegime.HIGH_VOLATILITY:
            position_mult = 0.5
            stop_mult = 2.0
        elif regime == MarketRegime.RANGING:
            position_mult = 0.8
            stop_mult = 1.2
        else:
            position_mult = 1.0
            stop_mult = 1.0
        
        signals = []
        
        if trend_type.startswith("strong"):
            signals.append(f"Strong {trend_type.split('_')[1]} trend detected")
        
        if volatility_state == "high":
            signals.append("High volatility regime - reduce position sizes")
        elif volatility_state == "low":
            signals.append("Low volatility - potential breakout setup")
        
        if self.garch.is_volatility_clustering():
            signals.append("Volatility clustering detected - expect continued volatility")
        
        if mr_score > 0.8:
            signals.append("Strong mean reversion opportunity")
        
        if self.current_regime_duration > 20:
            signals.append(f"Regime stable for {self.current_regime_duration} periods")
        
        transition_prob = self.hmm.get_transition_probability(
            self.hmm.current_state,
            (self.hmm.current_state + 1) % self.config.hmm_states
        )
        
        confidence = (hmm_confidence + trend_strength) / 2
        
        return RegimeState(
            regime=regime,
            confidence=confidence,
            trading_mode=trading_mode,
            volatility_state=volatility_state,
            trend_strength=trend_strength,
            momentum_score=momentum_score,
            mean_reversion_score=mr_score,
            regime_duration=self.current_regime_duration,
            transition_probability=transition_prob,
            recommended_position_size_mult=position_mult,
            recommended_stop_mult=stop_mult,
            signals=signals
        )
    
    def get_strategy_parameters(self, base_params: Dict[str, float]) -> Dict[str, float]:
        regime_state = self.detect_regime()
        
        adjusted_params = base_params.copy()
        
        adjusted_params["position_size"] = base_params.get("position_size", 1.0) * regime_state.recommended_position_size_mult
        adjusted_params["stop_loss"] = base_params.get("stop_loss", 0.02) * regime_state.recommended_stop_mult
        
        if regime_state.trading_mode == TradingMode.MOMENTUM:
            adjusted_params["trend_weight"] = base_params.get("trend_weight", 1.0) * 1.5
            adjusted_params["mean_reversion_weight"] = base_params.get("mean_reversion_weight", 1.0) * 0.5
        elif regime_state.trading_mode == TradingMode.MEAN_REVERSION:
            adjusted_params["trend_weight"] = base_params.get("trend_weight", 1.0) * 0.5
            adjusted_params["mean_reversion_weight"] = base_params.get("mean_reversion_weight", 1.0) * 1.5
        elif regime_state.trading_mode == TradingMode.BREAKOUT:
            adjusted_params["breakout_weight"] = base_params.get("breakout_weight", 1.0) * 2.0
        elif regime_state.trading_mode == TradingMode.DEFENSIVE:
            adjusted_params["position_size"] *= 0.5
            adjusted_params["min_signal_score"] = base_params.get("min_signal_score", 60) * 1.2
        
        return adjusted_params
    
    def forecast_regime(self, steps: int = 5) -> List[Dict[str, Any]]:
        vol_forecast = self.garch.forecast_volatility(steps)
        
        forecasts = []
        for i, vol in enumerate(vol_forecast):
            if vol > self.config.volatility_high_threshold:
                likely_regime = MarketRegime.HIGH_VOLATILITY
            else:
                likely_regime = self.regime_history[-1] if self.regime_history else MarketRegime.RANGING
            
            forecasts.append({
                "step": i + 1,
                "forecasted_volatility": vol,
                "likely_regime": likely_regime.value,
                "confidence": max(0.3, 1.0 - i * 0.15)
            })
        
        return forecasts
