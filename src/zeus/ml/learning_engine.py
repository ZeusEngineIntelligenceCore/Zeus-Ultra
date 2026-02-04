#!/usr/bin/env python3
"""
ZEUS AI LEARNING ENGINE - Continuous Improvement Through Machine Learning
Learns from trade outcomes to improve future decisions
"""

from __future__ import annotations
import json
import math
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import statistics

logger = logging.getLogger("Zeus.ML")


@dataclass
class TradeOutcome:
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    duration_seconds: int
    strategy: str
    confidence: float
    prebreakout_score: float
    entry_time: str
    exit_time: str
    exit_reason: str
    indicators_at_entry: Dict[str, float] = field(default_factory=dict)
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    

@dataclass
class SymbolProfile:
    symbol: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    best_entry_hour: int = 12
    best_hold_duration: int = 3600
    optimal_confidence_threshold: float = 75.0
    volatility_preference: float = 0.5
    last_updated: str = ""
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.wins / self.total_trades * 100


@dataclass 
class LearningState:
    symbol_profiles: Dict[str, SymbolProfile] = field(default_factory=dict)
    global_insights: Dict[str, Any] = field(default_factory=dict)
    strategy_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confidence_calibration: Dict[str, float] = field(default_factory=dict)
    time_of_day_performance: Dict[int, float] = field(default_factory=dict)
    trade_history_for_learning: List[Dict] = field(default_factory=list)
    model_version: str = "1.0"
    last_trained: str = ""
    total_learning_cycles: int = 0


class TradingLearningEngine:
    def __init__(self, data_file: str = "data/ml_learning.json"):
        self.data_file = Path(data_file)
        self.state = LearningState()
        self._load_state()
        
    def _load_state(self) -> None:
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    for symbol, profile_data in data.get("symbol_profiles", {}).items():
                        self.state.symbol_profiles[symbol] = SymbolProfile(**profile_data)
                    self.state.global_insights = data.get("global_insights", {})
                    self.state.strategy_performance = data.get("strategy_performance", {})
                    self.state.confidence_calibration = data.get("confidence_calibration", {})
                    self.state.time_of_day_performance = {
                        int(k): v for k, v in data.get("time_of_day_performance", {}).items()
                    }
                    self.state.trade_history_for_learning = data.get("trade_history_for_learning", [])
                    self.state.model_version = data.get("model_version", "1.0")
                    self.state.last_trained = data.get("last_trained", "")
                    self.state.total_learning_cycles = data.get("total_learning_cycles", 0)
                logger.info(f"Loaded ML state with {len(self.state.symbol_profiles)} symbol profiles")
            except Exception as e:
                logger.error(f"Failed to load ML state: {e}")
                
    def _save_state(self) -> None:
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = {
                "symbol_profiles": {k: asdict(v) for k, v in self.state.symbol_profiles.items()},
                "global_insights": self.state.global_insights,
                "strategy_performance": self.state.strategy_performance,
                "confidence_calibration": self.state.confidence_calibration,
                "time_of_day_performance": self.state.time_of_day_performance,
                "trade_history_for_learning": self.state.trade_history_for_learning[-500:],
                "model_version": self.state.model_version,
                "last_trained": self.state.last_trained,
                "total_learning_cycles": self.state.total_learning_cycles
            }
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save ML state: {e}")
            
    def record_trade_outcome(self, outcome: TradeOutcome) -> None:
        """Record a trade outcome for learning"""
        symbol = outcome.symbol
        
        if symbol not in self.state.symbol_profiles:
            self.state.symbol_profiles[symbol] = SymbolProfile(symbol=symbol)
            
        profile = self.state.symbol_profiles[symbol]
        profile.total_trades += 1
        profile.total_pnl += outcome.pnl
        profile.last_updated = datetime.now(timezone.utc).isoformat()
        
        if outcome.pnl >= 0:
            profile.wins += 1
            if profile.wins > 0:
                prev_avg = profile.avg_win_pct * (profile.wins - 1)
                profile.avg_win_pct = (prev_avg + outcome.pnl_pct) / profile.wins
        else:
            profile.losses += 1
            if profile.losses > 0:
                prev_avg = profile.avg_loss_pct * (profile.losses - 1)
                profile.avg_loss_pct = (prev_avg + abs(outcome.pnl_pct)) / profile.losses
                
        self.state.trade_history_for_learning.append(asdict(outcome))
        
        strategy = outcome.strategy
        if strategy not in self.state.strategy_performance:
            self.state.strategy_performance[strategy] = {
                "trades": 0, "wins": 0, "total_pnl": 0.0
            }
        self.state.strategy_performance[strategy]["trades"] += 1
        self.state.strategy_performance[strategy]["total_pnl"] += outcome.pnl
        if outcome.pnl >= 0:
            self.state.strategy_performance[strategy]["wins"] += 1
            
        try:
            entry_hour = datetime.fromisoformat(outcome.entry_time.replace('Z', '+00:00')).hour
            if entry_hour not in self.state.time_of_day_performance:
                self.state.time_of_day_performance[entry_hour] = 0.0
            current = self.state.time_of_day_performance[entry_hour]
            self.state.time_of_day_performance[entry_hour] = current * 0.9 + outcome.pnl_pct * 0.1
        except:
            pass
            
        self._save_state()
        logger.info(f"ML: Recorded trade outcome for {symbol}, PnL: ${outcome.pnl:.2f}")
        
    def run_learning_cycle(self) -> Dict[str, Any]:
        """Run a full learning cycle to update model parameters"""
        if len(self.state.trade_history_for_learning) < 5:
            return {"status": "insufficient_data", "trades_needed": 5 - len(self.state.trade_history_for_learning)}
            
        insights = {}
        
        winning_trades = [t for t in self.state.trade_history_for_learning if t.get("pnl", 0) >= 0]
        losing_trades = [t for t in self.state.trade_history_for_learning if t.get("pnl", 0) < 0]
        
        if winning_trades:
            avg_win_confidence = statistics.mean([t.get("confidence", 75) for t in winning_trades])
            insights["optimal_min_confidence"] = max(65, avg_win_confidence - 12)
            win_entry_prices = [t.get("entry_price", 0) for t in winning_trades]
            win_exit_prices = [t.get("exit_price", 0) for t in winning_trades]
            if win_entry_prices and win_exit_prices:
                avg_profit_pct = statistics.mean([
                    (exit - entry) / entry * 100 if entry > 0 else 0 
                    for entry, exit in zip(win_entry_prices, win_exit_prices)
                ])
                insights["avg_winning_profit_pct"] = round(avg_profit_pct, 2)
            
        if winning_trades and losing_trades:
            win_prebreakout_scores = [t.get("prebreakout_score", 0) for t in winning_trades]
            loss_prebreakout_scores = [t.get("prebreakout_score", 0) for t in losing_trades]
            
            if win_prebreakout_scores and loss_prebreakout_scores:
                avg_win_score = statistics.mean(win_prebreakout_scores)
                avg_loss_score = statistics.mean(loss_prebreakout_scores)
                insights["prebreakout_threshold"] = (avg_win_score + avg_loss_score) / 2 + 5
                
        if winning_trades:
            win_durations = [t.get("duration_seconds", 3600) for t in winning_trades]
            insights["optimal_hold_time"] = int(statistics.median(win_durations))
            
        best_strategies = []
        for strat, perf in self.state.strategy_performance.items():
            if perf["trades"] >= 5:
                win_rate = perf["wins"] / perf["trades"] * 100
                if win_rate >= 50:
                    best_strategies.append((strat, win_rate, perf["total_pnl"]))
        best_strategies.sort(key=lambda x: (x[1], x[2]), reverse=True)
        insights["best_strategies"] = [s[0] for s in best_strategies[:3]]
        
        best_hours = []
        for hour, avg_pnl in self.state.time_of_day_performance.items():
            if avg_pnl > 0:
                best_hours.append((hour, avg_pnl))
        best_hours.sort(key=lambda x: x[1], reverse=True)
        insights["best_trading_hours"] = [h[0] for h in best_hours[:8]]
        
        if len(self.state.trade_history_for_learning) >= 10:
            recent_trades = self.state.trade_history_for_learning[-20:]
            recent_wins = sum(1 for t in recent_trades if t.get("pnl", 0) >= 0)
            insights["recent_win_rate"] = round(recent_wins / len(recent_trades) * 100, 1)
            insights["recent_total_pnl"] = round(sum(t.get("pnl", 0) for t in recent_trades), 2)
            
            hold_times = [t.get("duration_seconds", 0) for t in self.state.trade_history_for_learning if t.get("pnl", 0) > 0]
            if hold_times:
                sorted_times = sorted(hold_times)
                idx = int(len(sorted_times) * 0.75)
                insights["optimal_max_hold_time"] = int(sorted_times[min(idx, len(sorted_times) - 1)])
        
        self.state.global_insights = insights
        self.state.last_trained = datetime.now(timezone.utc).isoformat()
        self.state.total_learning_cycles += 1
        self._save_state()
        
        logger.info(f"ML: Learning cycle complete. Insights: {insights}")
        return {"status": "success", "insights": insights, "cycle": self.state.total_learning_cycles}
        
    def get_trade_recommendation(
        self,
        symbol: str,
        confidence: float,
        prebreakout_score: float,
        strategy: str
    ) -> Tuple[bool, float, List[str]]:
        """Get ML-enhanced trade recommendation"""
        reasons = []
        adjusted_confidence = confidence
        should_trade = True
        
        if symbol in self.state.symbol_profiles:
            profile = self.state.symbol_profiles[symbol]
            if profile.total_trades >= 5:
                if profile.win_rate < 40:
                    adjusted_confidence *= 0.7
                    reasons.append(f"Low historical win rate ({profile.win_rate:.1f}%)")
                    if profile.win_rate < 25:
                        should_trade = False
                        reasons.append("Symbol blocked due to very low win rate")
                elif profile.win_rate > 70:
                    adjusted_confidence *= 1.15
                    reasons.append(f"High historical win rate ({profile.win_rate:.1f}%)")
                    
        insights = self.state.global_insights
        if insights.get("optimal_min_confidence"):
            if confidence < insights["optimal_min_confidence"]:
                adjusted_confidence *= 0.85
                reasons.append("Below learned optimal confidence threshold")
                
        if insights.get("prebreakout_threshold"):
            if prebreakout_score < insights["prebreakout_threshold"]:
                adjusted_confidence *= 0.9
                reasons.append("Below learned prebreakout threshold")
                
        if insights.get("best_strategies"):
            if strategy in insights["best_strategies"]:
                adjusted_confidence *= 1.1
                reasons.append(f"Using high-performing strategy: {strategy}")
                
        current_hour = datetime.now(timezone.utc).hour
        if insights.get("best_trading_hours"):
            if current_hour in insights["best_trading_hours"]:
                adjusted_confidence *= 1.05
                reasons.append("Trading during optimal hours")
            elif self.state.time_of_day_performance.get(current_hour, 0) < -1:
                adjusted_confidence *= 0.85
                reasons.append("Trading during historically poor hours")
                
        adjusted_confidence = min(100, max(0, adjusted_confidence))
        
        return should_trade, adjusted_confidence, reasons
        
    def get_symbol_insights(self, symbol: str) -> Dict[str, Any]:
        """Get learning insights for a specific symbol"""
        if symbol not in self.state.symbol_profiles:
            return {"status": "no_data", "symbol": symbol}
            
        profile = self.state.symbol_profiles[symbol]
        return {
            "symbol": symbol,
            "total_trades": profile.total_trades,
            "win_rate": profile.win_rate,
            "total_pnl": profile.total_pnl,
            "avg_win_pct": profile.avg_win_pct,
            "avg_loss_pct": profile.avg_loss_pct,
            "best_entry_hour": profile.best_entry_hour,
            "optimal_hold_duration": profile.best_hold_duration
        }
        
    def get_global_insights(self) -> Dict[str, Any]:
        """Get global learning insights"""
        return {
            "insights": self.state.global_insights,
            "learning_cycles": self.state.total_learning_cycles,
            "last_trained": self.state.last_trained,
            "symbols_tracked": len(self.state.symbol_profiles),
            "total_trades_learned": len(self.state.trade_history_for_learning),
            "strategy_performance": self.state.strategy_performance
        }
        
    def should_avoid_loss(
        self,
        symbol: str,
        current_pnl_pct: float,
        hold_time_seconds: int
    ) -> Tuple[bool, str]:
        """ML-driven loss avoidance recommendation"""
        if symbol in self.state.symbol_profiles:
            profile = self.state.symbol_profiles[symbol]
            if profile.losses > 0 and profile.avg_loss_pct > 0:
                if abs(current_pnl_pct) > profile.avg_loss_pct * 0.5:
                    return True, f"Approaching average loss threshold for {symbol}"
                    
        if current_pnl_pct < -2.0:
            return True, "Approaching maximum allowed loss threshold"
            
        optimal_hold = self.state.global_insights.get("optimal_hold_time", 7200)
        if hold_time_seconds > optimal_hold * 2 and current_pnl_pct < 0:
            return True, "Position held too long with negative P&L"
            
        return False, ""

    def extract_trade_features(self, trade_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract advanced features from trade data for pattern recognition"""
        features = {}
        indicators = trade_data.get("indicators_at_entry", {})
        features["rsi"] = indicators.get("rsi", 50) / 100
        features["macd_signal"] = 1 if indicators.get("macd_histogram", 0) > 0 else 0
        features["bb_position"] = indicators.get("bb_percent", 0.5)
        features["volume_ratio"] = min(indicators.get("volume_ratio", 1), 3) / 3
        features["atr_normalized"] = min(indicators.get("atr_pct", 2), 10) / 10
        entry_hour = 12
        try:
            entry_time = trade_data.get("entry_time", "")
            if entry_time:
                entry_hour = datetime.fromisoformat(entry_time.replace('Z', '+00:00')).hour
        except:
            pass
        features["hour_sin"] = math.sin(2 * math.pi * entry_hour / 24)
        features["hour_cos"] = math.cos(2 * math.pi * entry_hour / 24)
        features["prebreakout_score"] = trade_data.get("prebreakout_score", 50) / 100
        features["confidence"] = trade_data.get("confidence", 75) / 100
        duration = trade_data.get("duration_seconds", 3600)
        features["duration_norm"] = min(duration / 86400, 1)
        return features

    def predict_trade_outcome(
        self,
        symbol: str,
        confidence: float,
        prebreakout_score: float,
        indicators: Dict[str, float],
        current_hour: int
    ) -> Dict[str, Any]:
        """Predict expected trade outcome using learned patterns"""
        if len(self.state.trade_history_for_learning) < 10:
            return {
                "expected_pnl_pct": 0.0,
                "win_probability": 0.5,
                "confidence_level": "low",
                "recommendation": "insufficient_data"
            }
        winning_trades = [t for t in self.state.trade_history_for_learning if t.get("pnl", 0) > 0]
        losing_trades = [t for t in self.state.trade_history_for_learning if t.get("pnl", 0) <= 0]
        win_rate = len(winning_trades) / len(self.state.trade_history_for_learning)
        base_win_prob = win_rate
        if symbol in self.state.symbol_profiles:
            profile = self.state.symbol_profiles[symbol]
            if profile.total_trades >= 3:
                symbol_wr = profile.win_rate / 100
                base_win_prob = (base_win_prob + symbol_wr) / 2
        confidence_boost = (confidence - 70) / 100 * 0.15
        prebreakout_boost = (prebreakout_score - 50) / 100 * 0.2
        rsi = indicators.get("rsi", 50)
        if 30 < rsi < 70:
            rsi_boost = 0.05
        elif rsi <= 30 or rsi >= 70:
            rsi_boost = 0.08
        else:
            rsi_boost = 0
        hour_perf = self.state.time_of_day_performance.get(current_hour, 0)
        time_boost = hour_perf * 0.02 if hour_perf > 0 else hour_perf * 0.03
        adjusted_win_prob = min(0.95, max(0.05, base_win_prob + confidence_boost + prebreakout_boost + rsi_boost + time_boost))
        if winning_trades:
            avg_win = statistics.mean([t.get("pnl_pct", 3) for t in winning_trades])
        else:
            avg_win = 3.0
        if losing_trades:
            avg_loss = statistics.mean([abs(t.get("pnl_pct", 2)) for t in losing_trades])
        else:
            avg_loss = 2.0
        expected_pnl = (adjusted_win_prob * avg_win) - ((1 - adjusted_win_prob) * avg_loss)
        if adjusted_win_prob >= 0.7 and expected_pnl > 1:
            recommendation = "strong_buy"
            confidence_level = "high"
        elif adjusted_win_prob >= 0.55 and expected_pnl > 0:
            recommendation = "buy"
            confidence_level = "medium"
        elif adjusted_win_prob < 0.4 or expected_pnl < -1:
            recommendation = "avoid"
            confidence_level = "high"
        else:
            recommendation = "neutral"
            confidence_level = "low"
        return {
            "expected_pnl_pct": round(expected_pnl, 2),
            "win_probability": round(adjusted_win_prob, 3),
            "confidence_level": confidence_level,
            "recommendation": recommendation,
            "avg_win_pct": round(avg_win, 2),
            "avg_loss_pct": round(avg_loss, 2),
            "factors": {
                "base_win_rate": round(win_rate, 3),
                "confidence_boost": round(confidence_boost, 3),
                "prebreakout_boost": round(prebreakout_boost, 3),
                "time_boost": round(time_boost, 3)
            }
        }

    def detect_pattern_clusters(self) -> Dict[str, Any]:
        """Identify winning and losing pattern clusters from trade history"""
        if len(self.state.trade_history_for_learning) < 20:
            return {"status": "insufficient_data"}
        winning_patterns = []
        losing_patterns = []
        for trade in self.state.trade_history_for_learning:
            features = self.extract_trade_features(trade)
            if trade.get("pnl", 0) > 0:
                winning_patterns.append(features)
            else:
                losing_patterns.append(features)
        if not winning_patterns or not losing_patterns:
            return {"status": "insufficient_data"}
        win_centroids = {}
        lose_centroids = {}
        feature_keys = winning_patterns[0].keys()
        for key in feature_keys:
            win_values = [p[key] for p in winning_patterns]
            lose_values = [p[key] for p in losing_patterns]
            win_centroids[key] = statistics.mean(win_values) if win_values else 0.5
            lose_centroids[key] = statistics.mean(lose_values) if lose_values else 0.5
        discriminating_features = []
        for key in feature_keys:
            diff = abs(win_centroids[key] - lose_centroids[key])
            if diff > 0.1:
                discriminating_features.append({
                    "feature": key,
                    "win_avg": round(win_centroids[key], 3),
                    "lose_avg": round(lose_centroids[key], 3),
                    "difference": round(diff, 3)
                })
        discriminating_features.sort(key=lambda x: x["difference"], reverse=True)
        return {
            "status": "success",
            "winning_pattern_count": len(winning_patterns),
            "losing_pattern_count": len(losing_patterns),
            "win_centroids": {k: round(v, 3) for k, v in win_centroids.items()},
            "lose_centroids": {k: round(v, 3) for k, v in lose_centroids.items()},
            "top_discriminating_features": discriminating_features[:5]
        }

    def get_adaptive_confidence_threshold(self, symbol: str, market_volatility: float = 0.5) -> float:
        """Calculate adaptive confidence threshold based on recent performance and market conditions"""
        base_threshold = 70.0
        recent_trades = self.state.trade_history_for_learning[-20:]
        if recent_trades:
            recent_wins = sum(1 for t in recent_trades if t.get("pnl", 0) > 0)
            recent_wr = recent_wins / len(recent_trades)
            if recent_wr > 0.65:
                base_threshold -= 5
            elif recent_wr < 0.4:
                base_threshold += 8
        if symbol in self.state.symbol_profiles:
            profile = self.state.symbol_profiles[symbol]
            if profile.win_rate > 70:
                base_threshold -= 5
            elif profile.win_rate < 40:
                base_threshold += 10
        if market_volatility > 0.7:
            base_threshold += 5
        elif market_volatility < 0.3:
            base_threshold -= 3
        return max(60, min(90, base_threshold))

    def calculate_position_score(
        self,
        symbol: str,
        confidence: float,
        prebreakout_score: float,
        rsi: float,
        volume_ratio: float
    ) -> Dict[str, Any]:
        """Calculate comprehensive position score combining all ML factors"""
        prediction = self.predict_trade_outcome(
            symbol, confidence, prebreakout_score,
            {"rsi": rsi}, datetime.now(timezone.utc).hour
        )
        base_score = confidence * 0.3 + prebreakout_score * 0.25
        if 30 < rsi < 70:
            rsi_score = 10
        elif rsi <= 30:
            rsi_score = 15
        elif rsi >= 70:
            rsi_score = 5
        else:
            rsi_score = 8
        volume_score = min(volume_ratio * 5, 15)
        ml_score = prediction["win_probability"] * 20
        symbol_score = 0
        if symbol in self.state.symbol_profiles:
            profile = self.state.symbol_profiles[symbol]
            if profile.win_rate > 60:
                symbol_score = 10
            elif profile.win_rate > 50:
                symbol_score = 5
            elif profile.win_rate < 40:
                symbol_score = -10
        total_score = base_score + rsi_score + volume_score + ml_score + symbol_score
        return {
            "total_score": round(total_score, 1),
            "breakdown": {
                "base": round(base_score, 1),
                "rsi": rsi_score,
                "volume": round(volume_score, 1),
                "ml": round(ml_score, 1),
                "symbol_history": symbol_score
            },
            "prediction": prediction,
            "recommendation": "strong" if total_score >= 80 else "moderate" if total_score >= 60 else "weak"
        }
