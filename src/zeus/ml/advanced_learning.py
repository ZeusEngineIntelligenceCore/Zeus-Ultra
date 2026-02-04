#!/usr/bin/env python3
"""
ZEUS ADVANCED ML LEARNING ENGINE
Enhanced machine learning with pattern recognition, market regime detection,
reinforcement learning, and adaptive position sizing.
"""

from __future__ import annotations
import json
import math
import logging
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import statistics

logger = logging.getLogger("Zeus.AdvancedML")


@dataclass
class MarketRegime:
    regime_type: str  # TRENDING_UP, TRENDING_DOWN, RANGING, HIGH_VOLATILITY, LOW_VOLATILITY
    confidence: float
    detected_at: str
    btc_correlation: float = 0.0
    volatility_percentile: float = 0.5
    trend_strength: float = 0.0


@dataclass
class TradeFeatureVector:
    symbol: str
    timestamp: str
    
    rsi_14: float = 50.0
    rsi_7: float = 50.0
    macd_histogram: float = 0.0
    macd_signal: float = 0.0
    bb_percent: float = 0.5
    bb_width: float = 0.02
    
    volume_ratio: float = 1.0
    volume_trend: float = 0.0  # positive = increasing, negative = decreasing
    
    atr_percent: float = 2.0
    volatility_rank: float = 0.5  # 0-1 percentile vs recent history
    
    trend_strength: float = 0.0  # ADX-like metric
    trend_direction: float = 0.0  # -1 to 1
    
    support_distance: float = 0.0  # % distance from nearest support
    resistance_distance: float = 0.0  # % distance from nearest resistance
    
    prebreakout_score: float = 0.0
    confidence: float = 75.0
    
    hour_of_day: int = 12
    day_of_week: int = 0
    
    market_regime: str = "UNKNOWN"
    btc_trend: float = 0.0  # BTC direction (-1 to 1)
    fear_greed: int = 50
    
    mtf_alignment: float = 0.0  # multi-timeframe alignment score
    
    def to_vector(self) -> List[float]:
        """Convert to normalized feature vector for similarity calculations"""
        return [
            self.rsi_14 / 100,
            self.rsi_7 / 100,
            max(-1, min(1, self.macd_histogram)),
            self.bb_percent,
            min(self.bb_width * 10, 1),
            min(self.volume_ratio / 3, 1),
            max(-1, min(1, self.volume_trend)),
            min(self.atr_percent / 10, 1),
            self.volatility_rank,
            max(-1, min(1, self.trend_strength)),
            max(-1, min(1, self.trend_direction)),
            min(abs(self.support_distance) / 5, 1),
            min(abs(self.resistance_distance) / 5, 1),
            self.prebreakout_score / 100,
            self.confidence / 100,
            math.sin(2 * math.pi * self.hour_of_day / 24),
            math.cos(2 * math.pi * self.hour_of_day / 24),
            math.sin(2 * math.pi * self.day_of_week / 7),
            max(-1, min(1, self.btc_trend)),
            self.fear_greed / 100,
            self.mtf_alignment
        ]


@dataclass
class ReinforcementState:
    exploration_rate: float = 0.15  # epsilon for exploration vs exploitation
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    action_values: Dict[str, float] = field(default_factory=dict)  # Q-values
    state_action_counts: Dict[str, int] = field(default_factory=dict)
    recent_rewards: List[float] = field(default_factory=list)
    cumulative_reward: float = 0.0
    episodes: int = 0


@dataclass
class AdvancedLearningState:
    feature_vectors: List[Dict] = field(default_factory=list)
    winning_patterns: List[Dict] = field(default_factory=list)
    losing_patterns: List[Dict] = field(default_factory=list)
    
    market_regimes: List[Dict] = field(default_factory=list)
    current_regime: Optional[Dict] = None
    
    reinforcement: ReinforcementState = field(default_factory=ReinforcementState)
    
    position_sizing_model: Dict[str, float] = field(default_factory=dict)
    
    indicator_importance: Dict[str, float] = field(default_factory=dict)
    optimal_thresholds: Dict[str, float] = field(default_factory=dict)
    
    correlation_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    model_version: str = "2.0"
    last_updated: str = ""
    training_epochs: int = 0


class AdvancedLearningEngine:
    """Enhanced ML engine with pattern recognition and reinforcement learning"""
    
    def __init__(self, data_file: str = "data/advanced_ml.json"):
        self.data_file = Path(data_file)
        self.state = AdvancedLearningState()
        self._load_state()
        
    def _load_state(self) -> None:
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    
                self.state.feature_vectors = data.get("feature_vectors", [])
                self.state.winning_patterns = data.get("winning_patterns", [])
                self.state.losing_patterns = data.get("losing_patterns", [])
                self.state.market_regimes = data.get("market_regimes", [])
                self.state.current_regime = data.get("current_regime")
                
                rl_data = data.get("reinforcement", {})
                self.state.reinforcement = ReinforcementState(
                    exploration_rate=rl_data.get("exploration_rate", 0.15),
                    learning_rate=rl_data.get("learning_rate", 0.1),
                    discount_factor=rl_data.get("discount_factor", 0.95),
                    action_values=rl_data.get("action_values", {}),
                    state_action_counts=rl_data.get("state_action_counts", {}),
                    recent_rewards=rl_data.get("recent_rewards", []),
                    cumulative_reward=rl_data.get("cumulative_reward", 0.0),
                    episodes=rl_data.get("episodes", 0)
                )
                
                self.state.position_sizing_model = data.get("position_sizing_model", {})
                self.state.indicator_importance = data.get("indicator_importance", {})
                self.state.optimal_thresholds = data.get("optimal_thresholds", {})
                self.state.correlation_matrix = data.get("correlation_matrix", {})
                self.state.model_version = data.get("model_version", "2.0")
                self.state.last_updated = data.get("last_updated", "")
                self.state.training_epochs = data.get("training_epochs", 0)
                
                logger.info(f"Loaded advanced ML state with {len(self.state.feature_vectors)} patterns")
            except Exception as e:
                logger.error(f"Failed to load advanced ML state: {e}")
                
    def _save_state(self) -> None:
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = {
                "feature_vectors": self.state.feature_vectors[-1000:],
                "winning_patterns": self.state.winning_patterns[-500:],
                "losing_patterns": self.state.losing_patterns[-500:],
                "market_regimes": self.state.market_regimes[-100:],
                "current_regime": self.state.current_regime,
                "reinforcement": asdict(self.state.reinforcement),
                "position_sizing_model": self.state.position_sizing_model,
                "indicator_importance": self.state.indicator_importance,
                "optimal_thresholds": self.state.optimal_thresholds,
                "correlation_matrix": self.state.correlation_matrix,
                "model_version": self.state.model_version,
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "training_epochs": self.state.training_epochs
            }
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save advanced ML state: {e}")
            
    def detect_market_regime(
        self,
        closes: List[float],
        highs: List[float],
        lows: List[float],
        volumes: List[float],
        btc_closes: Optional[List[float]] = None
    ) -> MarketRegime:
        """Detect current market regime using multiple indicators"""
        if len(closes) < 50:
            return MarketRegime(
                regime_type="UNKNOWN",
                confidence=0.0,
                detected_at=datetime.now(timezone.utc).isoformat()
            )
            
        sma_20 = sum(closes[-20:]) / 20
        sma_50 = sum(closes[-50:]) / 50
        current_price = closes[-1]
        
        price_vs_sma20 = (current_price - sma_20) / sma_20 * 100
        sma_slope = (sma_20 - sum(closes[-25:-5]) / 20) / sma_20 * 100
        
        tr_values = []
        for i in range(1, min(14, len(closes))):
            tr = max(
                highs[-i] - lows[-i],
                abs(highs[-i] - closes[-i-1]),
                abs(lows[-i] - closes[-i-1])
            )
            tr_values.append(tr)
        atr = sum(tr_values) / len(tr_values) if tr_values else 0
        atr_percent = (atr / current_price * 100) if current_price > 0 else 0
        
        all_atrs = []
        for i in range(min(50, len(closes) - 14)):
            period_tr = []
            for j in range(14):
                idx = -50 + i + j
                if idx < 0 and abs(idx) < len(closes):
                    tr = max(
                        highs[idx] - lows[idx],
                        abs(highs[idx] - closes[idx-1]),
                        abs(lows[idx] - closes[idx-1])
                    )
                    period_tr.append(tr)
            if period_tr:
                all_atrs.append(sum(period_tr) / len(period_tr))
        
        volatility_percentile = 0.5
        if all_atrs:
            sorted_atrs = sorted(all_atrs)
            vol_rank = sum(1 for a in sorted_atrs if a <= atr) / len(sorted_atrs)
            volatility_percentile = vol_rank
        
        btc_correlation = 0.0
        if btc_closes and len(btc_closes) >= 20:
            try:
                returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(-19, 0)]
                btc_returns = [(btc_closes[i] - btc_closes[i-1]) / btc_closes[i-1] for i in range(-19, 0)]
                
                mean_r = sum(returns) / len(returns)
                mean_btc = sum(btc_returns) / len(btc_returns)
                
                cov = sum((r - mean_r) * (b - mean_btc) for r, b in zip(returns, btc_returns)) / len(returns)
                std_r = (sum((r - mean_r) ** 2 for r in returns) / len(returns)) ** 0.5
                std_btc = (sum((b - mean_btc) ** 2 for b in btc_returns) / len(btc_returns)) ** 0.5
                
                if std_r > 0 and std_btc > 0:
                    btc_correlation = cov / (std_r * std_btc)
            except:
                pass
        
        if sma_slope > 1.5 and price_vs_sma20 > 2:
            regime_type = "TRENDING_UP"
            confidence = min(0.9, 0.5 + abs(sma_slope) / 5)
        elif sma_slope < -1.5 and price_vs_sma20 < -2:
            regime_type = "TRENDING_DOWN"
            confidence = min(0.9, 0.5 + abs(sma_slope) / 5)
        elif volatility_percentile > 0.8:
            regime_type = "HIGH_VOLATILITY"
            confidence = volatility_percentile
        elif volatility_percentile < 0.2:
            regime_type = "LOW_VOLATILITY"
            confidence = 1 - volatility_percentile
        else:
            regime_type = "RANGING"
            confidence = 0.6
            
        regime = MarketRegime(
            regime_type=regime_type,
            confidence=confidence,
            detected_at=datetime.now(timezone.utc).isoformat(),
            btc_correlation=btc_correlation,
            volatility_percentile=volatility_percentile,
            trend_strength=abs(sma_slope)
        )
        
        self.state.current_regime = asdict(regime)
        self.state.market_regimes.append(asdict(regime))
        self._save_state()
        
        return regime
        
    def record_trade_with_features(
        self,
        features: TradeFeatureVector,
        pnl: float,
        pnl_pct: float,
        duration_seconds: int,
        exit_reason: str
    ) -> None:
        """Record a trade with full feature vector for pattern learning"""
        trade_record = {
            "features": asdict(features),
            "outcome": {
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "duration_seconds": duration_seconds,
                "exit_reason": exit_reason,
                "is_winner": pnl > 0
            },
            "recorded_at": datetime.now(timezone.utc).isoformat()
        }
        
        self.state.feature_vectors.append(trade_record)
        
        if pnl > 0:
            self.state.winning_patterns.append(trade_record)
        else:
            self.state.losing_patterns.append(trade_record)
            
        self._update_reinforcement_learning(features, pnl_pct)
        self._update_indicator_importance()
        
        self._save_state()
        logger.info(f"Advanced ML: Recorded trade with {len(features.to_vector())} features")
        
    def _update_reinforcement_learning(
        self,
        features: TradeFeatureVector,
        reward: float
    ) -> None:
        """Update Q-values using trade outcome as reward"""
        state_key = self._features_to_state_key(features)
        action_key = f"{state_key}:trade"
        
        current_q = self.state.reinforcement.action_values.get(action_key, 0.0)
        visit_count = self.state.reinforcement.state_action_counts.get(action_key, 0) + 1
        
        adaptive_lr = self.state.reinforcement.learning_rate / (1 + visit_count * 0.01)
        
        new_q = current_q + adaptive_lr * (reward - current_q)
        
        self.state.reinforcement.action_values[action_key] = new_q
        self.state.reinforcement.state_action_counts[action_key] = visit_count
        self.state.reinforcement.recent_rewards.append(reward)
        self.state.reinforcement.recent_rewards = self.state.reinforcement.recent_rewards[-100:]
        self.state.reinforcement.cumulative_reward += reward
        self.state.reinforcement.episodes += 1
        
        if self.state.reinforcement.episodes % 10 == 0:
            self.state.reinforcement.exploration_rate = max(
                0.05,
                self.state.reinforcement.exploration_rate * 0.99
            )
            
    def _features_to_state_key(self, features: TradeFeatureVector) -> str:
        """Convert features to a discretized state key for RL"""
        rsi_bucket = "oversold" if features.rsi_14 < 30 else "overbought" if features.rsi_14 > 70 else "neutral"
        vol_bucket = "high" if features.volatility_rank > 0.7 else "low" if features.volatility_rank < 0.3 else "med"
        trend_bucket = "up" if features.trend_direction > 0.3 else "down" if features.trend_direction < -0.3 else "flat"
        regime = features.market_regime[:3].lower() if features.market_regime else "unk"
        
        return f"{rsi_bucket}_{vol_bucket}_{trend_bucket}_{regime}"
        
    def _update_indicator_importance(self) -> None:
        """Calculate which indicators best discriminate winners from losers"""
        if len(self.state.winning_patterns) < 5 or len(self.state.losing_patterns) < 5:
            return
            
        importance = {}
        
        indicator_keys = [
            "rsi_14", "rsi_7", "macd_histogram", "bb_percent", "bb_width",
            "volume_ratio", "volume_trend", "atr_percent", "volatility_rank",
            "trend_strength", "trend_direction", "prebreakout_score", "confidence",
            "fear_greed", "mtf_alignment"
        ]
        
        for key in indicator_keys:
            try:
                win_values = [p["features"].get(key, 0) for p in self.state.winning_patterns[-50:]]
                lose_values = [p["features"].get(key, 0) for p in self.state.losing_patterns[-50:]]
                
                if win_values and lose_values:
                    win_mean = statistics.mean(win_values)
                    lose_mean = statistics.mean(lose_values)
                    
                    combined = win_values + lose_values
                    combined_std = statistics.stdev(combined) if len(combined) > 1 else 1
                    
                    if combined_std > 0:
                        separation = abs(win_mean - lose_mean) / combined_std
                        importance[key] = round(separation, 4)
            except:
                pass
                
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        self.state.indicator_importance = sorted_importance
        
    def calculate_pattern_similarity(
        self,
        current_features: TradeFeatureVector,
        top_n: int = 5
    ) -> Dict[str, Any]:
        """Find historical trades most similar to current setup"""
        if not self.state.feature_vectors:
            return {"status": "no_data", "similar_trades": []}
            
        current_vector = current_features.to_vector()
        
        similarities = []
        
        for trade_record in self.state.feature_vectors[-200:]:
            try:
                features_dict = trade_record.get("features", {})
                
                historical = TradeFeatureVector(
                    symbol=features_dict.get("symbol", ""),
                    timestamp=features_dict.get("timestamp", ""),
                    rsi_14=features_dict.get("rsi_14", 50),
                    rsi_7=features_dict.get("rsi_7", 50),
                    macd_histogram=features_dict.get("macd_histogram", 0),
                    bb_percent=features_dict.get("bb_percent", 0.5),
                    bb_width=features_dict.get("bb_width", 0.02),
                    volume_ratio=features_dict.get("volume_ratio", 1),
                    volume_trend=features_dict.get("volume_trend", 0),
                    atr_percent=features_dict.get("atr_percent", 2),
                    volatility_rank=features_dict.get("volatility_rank", 0.5),
                    trend_strength=features_dict.get("trend_strength", 0),
                    trend_direction=features_dict.get("trend_direction", 0),
                    support_distance=features_dict.get("support_distance", 0),
                    resistance_distance=features_dict.get("resistance_distance", 0),
                    prebreakout_score=features_dict.get("prebreakout_score", 0),
                    confidence=features_dict.get("confidence", 75),
                    hour_of_day=features_dict.get("hour_of_day", 12),
                    day_of_week=features_dict.get("day_of_week", 0),
                    market_regime=features_dict.get("market_regime", "UNKNOWN"),
                    btc_trend=features_dict.get("btc_trend", 0),
                    fear_greed=features_dict.get("fear_greed", 50),
                    mtf_alignment=features_dict.get("mtf_alignment", 0)
                )
                
                hist_vector = historical.to_vector()
                
                diff_sum = sum((a - b) ** 2 for a, b in zip(current_vector, hist_vector))
                distance = math.sqrt(diff_sum)
                similarity = 1 / (1 + distance)
                
                weighted_diff = 0
                for i, key in enumerate(self.state.indicator_importance.keys()):
                    if i < len(current_vector) and i < len(hist_vector):
                        weight = self.state.indicator_importance.get(key, 1.0)
                        weighted_diff += weight * (current_vector[i] - hist_vector[i]) ** 2
                weighted_similarity = 1 / (1 + math.sqrt(weighted_diff))
                
                similarities.append({
                    "trade": trade_record,
                    "similarity": similarity,
                    "weighted_similarity": weighted_similarity,
                    "outcome": trade_record.get("outcome", {})
                })
            except Exception as e:
                continue
                
        similarities.sort(key=lambda x: x["weighted_similarity"], reverse=True)
        top_similar = similarities[:top_n]
        
        if top_similar:
            win_count = sum(1 for s in top_similar if s["outcome"].get("is_winner", False))
            avg_pnl = statistics.mean([s["outcome"].get("pnl_pct", 0) for s in top_similar])
            
            return {
                "status": "success",
                "similar_trades": [
                    {
                        "symbol": s["trade"]["features"].get("symbol"),
                        "similarity": round(s["weighted_similarity"], 3),
                        "pnl_pct": round(s["outcome"].get("pnl_pct", 0), 2),
                        "was_winner": s["outcome"].get("is_winner", False)
                    }
                    for s in top_similar
                ],
                "pattern_win_rate": round(win_count / len(top_similar) * 100, 1),
                "expected_pnl_pct": round(avg_pnl, 2)
            }
        
        return {"status": "no_similar_trades", "similar_trades": []}
        
    def get_adaptive_position_size(
        self,
        base_size: float,
        features: TradeFeatureVector,
        account_equity: float,
        max_risk_pct: float = 2.0
    ) -> Dict[str, Any]:
        """Calculate ML-optimized position size"""
        multiplier = 1.0
        reasons = []
        
        similarity = self.calculate_pattern_similarity(features, top_n=10)
        if similarity.get("status") == "success":
            pattern_wr = similarity.get("pattern_win_rate", 50)
            if pattern_wr > 70:
                multiplier *= 1.3
                reasons.append(f"High pattern similarity win rate ({pattern_wr}%)")
            elif pattern_wr < 40:
                multiplier *= 0.6
                reasons.append(f"Low pattern similarity win rate ({pattern_wr}%)")
        
        state_key = self._features_to_state_key(features)
        action_key = f"{state_key}:trade"
        q_value = self.state.reinforcement.action_values.get(action_key, 0)
        
        if q_value > 2:
            multiplier *= 1.2
            reasons.append(f"Strong Q-value ({q_value:.2f}) for this setup")
        elif q_value < -1:
            multiplier *= 0.7
            reasons.append(f"Negative Q-value ({q_value:.2f}) for this setup")
        
        regime = self.state.current_regime
        if regime:
            regime_type = regime.get("regime_type", "UNKNOWN")
            if regime_type == "HIGH_VOLATILITY":
                multiplier *= 0.7
                reasons.append("Reduced size due to high volatility regime")
            elif regime_type == "TRENDING_UP" and features.trend_direction > 0:
                multiplier *= 1.15
                reasons.append("Increased size in uptrend alignment")
        
        if features.confidence > 90:
            multiplier *= 1.1
            reasons.append("High signal confidence")
        elif features.confidence < 70:
            multiplier *= 0.85
            reasons.append("Lower signal confidence")
        
        final_size = base_size * min(2.0, max(0.3, multiplier))
        
        max_position = account_equity * max_risk_pct / 100
        final_size = min(final_size, max_position)
        
        return {
            "recommended_size": round(final_size, 4),
            "size_multiplier": round(multiplier, 3),
            "reasons": reasons,
            "pattern_win_rate": similarity.get("pattern_win_rate") if similarity.get("status") == "success" else None,
            "q_value": round(q_value, 3)
        }
        
    def should_take_trade(
        self,
        features: TradeFeatureVector,
        min_confidence: float = 70.0
    ) -> Tuple[bool, float, List[str]]:
        """ML-enhanced trade decision with reinforcement learning"""
        reasons = []
        score = 50.0
        
        if features.confidence >= min_confidence:
            score += 10
        else:
            score -= 15
            reasons.append(f"Confidence {features.confidence:.0f}% below threshold {min_confidence:.0f}%")
        
        similarity = self.calculate_pattern_similarity(features, top_n=10)
        if similarity.get("status") == "success":
            pattern_wr = similarity.get("pattern_win_rate", 50)
            expected_pnl = similarity.get("expected_pnl_pct", 0)
            
            if pattern_wr > 65 and expected_pnl > 1:
                score += 20
                reasons.append(f"Similar patterns: {pattern_wr}% win rate, {expected_pnl}% expected PnL")
            elif pattern_wr < 40 or expected_pnl < -1:
                score -= 25
                reasons.append(f"Warning: Similar patterns show {pattern_wr}% win rate")
        
        state_key = self._features_to_state_key(features)
        action_key = f"{state_key}:trade"
        q_value = self.state.reinforcement.action_values.get(action_key, 0)
        
        if q_value > 1.5:
            score += 15
            reasons.append(f"RL recommends: Q-value {q_value:.2f}")
        elif q_value < -1:
            score -= 20
            reasons.append(f"RL warns: Q-value {q_value:.2f}")
        
        regime = self.state.current_regime
        if regime:
            regime_type = regime.get("regime_type", "UNKNOWN")
            if regime_type == "HIGH_VOLATILITY":
                score -= 10
                reasons.append("Caution: High volatility regime")
            elif regime_type == "TRENDING_UP" and features.trend_direction > 0.3:
                score += 10
                reasons.append("Aligned with uptrend regime")
            elif regime_type == "TRENDING_DOWN" and features.trend_direction < -0.3:
                score -= 15
                reasons.append("Against uptrend preference in downtrend")
        
        top_indicators = list(self.state.indicator_importance.keys())[:5]
        for indicator in top_indicators:
            if indicator == "prebreakout_score" and features.prebreakout_score > 80:
                score += 5
            elif indicator == "volume_ratio" and features.volume_ratio > 1.5:
                score += 5
            elif indicator == "mtf_alignment" and features.mtf_alignment > 0.6:
                score += 5
        
        should_trade = score >= 60
        confidence = min(100, max(0, score))
        
        return should_trade, confidence, reasons
        
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get comprehensive learning summary"""
        rl = self.state.reinforcement
        
        recent_performance = None
        if rl.recent_rewards:
            recent_performance = {
                "avg_reward": round(statistics.mean(rl.recent_rewards), 3),
                "win_rate": round(sum(1 for r in rl.recent_rewards if r > 0) / len(rl.recent_rewards) * 100, 1),
                "best_reward": round(max(rl.recent_rewards), 3),
                "worst_reward": round(min(rl.recent_rewards), 3)
            }
        
        return {
            "total_patterns_learned": len(self.state.feature_vectors),
            "winning_patterns": len(self.state.winning_patterns),
            "losing_patterns": len(self.state.losing_patterns),
            "win_ratio": round(len(self.state.winning_patterns) / max(1, len(self.state.feature_vectors)) * 100, 1),
            "rl_episodes": rl.episodes,
            "rl_exploration_rate": round(rl.exploration_rate, 3),
            "rl_cumulative_reward": round(rl.cumulative_reward, 2),
            "recent_performance": recent_performance,
            "top_indicators": list(self.state.indicator_importance.keys())[:5],
            "current_regime": self.state.current_regime,
            "model_version": self.state.model_version,
            "last_updated": self.state.last_updated
        }
        
    def run_optimization_cycle(self) -> Dict[str, Any]:
        """Run optimization to improve thresholds and weights"""
        if len(self.state.feature_vectors) < 20:
            return {"status": "insufficient_data", "minimum_needed": 20}
            
        optimizations = {}
        
        self._update_indicator_importance()
        optimizations["indicator_importance_updated"] = True
        
        winning = [p["features"] for p in self.state.winning_patterns[-100:]]
        if winning:
            for key in ["rsi_14", "prebreakout_score", "confidence", "volume_ratio"]:
                values = [w.get(key, 0) for w in winning]
                if values:
                    self.state.optimal_thresholds[f"min_{key}"] = round(statistics.mean(values) - statistics.stdev(values) if len(values) > 1 else statistics.mean(values), 2)
                    self.state.optimal_thresholds[f"optimal_{key}"] = round(statistics.mean(values), 2)
        optimizations["thresholds_updated"] = True
        
        if self.state.reinforcement.recent_rewards:
            recent_wr = sum(1 for r in self.state.reinforcement.recent_rewards if r > 0) / len(self.state.reinforcement.recent_rewards)
            if recent_wr > 0.65:
                self.state.reinforcement.exploration_rate = max(0.05, self.state.reinforcement.exploration_rate - 0.02)
            elif recent_wr < 0.4:
                self.state.reinforcement.exploration_rate = min(0.25, self.state.reinforcement.exploration_rate + 0.03)
        optimizations["exploration_rate_adjusted"] = self.state.reinforcement.exploration_rate
        
        self.state.training_epochs += 1
        self._save_state()
        
        return {
            "status": "success",
            "optimizations": optimizations,
            "epoch": self.state.training_epochs
        }
