#!/usr/bin/env python3
"""
ZEUS MULTI-TIMEFRAME FUSION SYSTEM
Combines all indicators and KPIs across 5m/15m/1h/4h/1d timeframes
to create ultra-accurate, competition-crushing trading signals.
Uses weighted consensus, divergence detection, and ML-enhanced scoring.
"""

import logging
import statistics
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone
from enum import Enum
import numpy as np

logger = logging.getLogger("Zeus.MTFFusion")


class TimeframeWeight(Enum):
    TF_5M = 0.10
    TF_15M = 0.15
    TF_1H = 0.25
    TF_4H = 0.30
    TF_1D = 0.20


@dataclass
class TimeframeIndicators:
    timeframe: str
    rsi: float
    macd: float
    macd_signal: float
    macd_histogram: float
    ema_9: float
    ema_21: float
    ema_50: float
    sma_200: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_width: float
    atr: float
    adx: float
    plus_di: float
    minus_di: float
    stoch_k: float
    stoch_d: float
    cci: float
    mfi: float
    obv_trend: float
    vwap: float
    current_price: float
    volume: float
    volume_sma: float
    
    schaff_tc: Optional[float] = None
    fisher: Optional[float] = None
    dpo: Optional[float] = None
    kst: Optional[float] = None
    cmo: Optional[float] = None
    rvi: Optional[float] = None
    ppo: Optional[float] = None
    bop: Optional[float] = None
    pmo: Optional[float] = None
    
    hurst: Optional[float] = None
    entropy: Optional[float] = None
    fractal_high: Optional[float] = None
    fractal_low: Optional[float] = None


@dataclass
class TimeframeScore:
    timeframe: str
    trend_score: float
    momentum_score: float
    volatility_score: float
    volume_score: float
    reversal_probability: float
    continuation_probability: float
    overall_score: float
    signal_direction: str
    confidence: float
    key_signals: List[str]


@dataclass
class MTFConsensus:
    bullish_count: int
    bearish_count: int
    neutral_count: int
    consensus_direction: str
    consensus_strength: float
    divergences: List[str]
    alignment_score: float


@dataclass
class FusedSignal:
    symbol: str
    timestamp: datetime
    direction: str
    strength: float
    confidence: float
    
    entry_score: float
    timing_score: float
    risk_score: float
    
    timeframe_scores: Dict[str, TimeframeScore]
    consensus: MTFConsensus
    
    primary_drivers: List[str]
    warnings: List[str]
    
    recommended_action: str
    optimal_entry_zone: Tuple[float, float]
    stop_loss_zone: Tuple[float, float]
    target_zones: List[Tuple[float, float]]
    
    overall_grade: str
    beats_competition_score: float


class MultiTimeframeFusion:
    def __init__(self):
        self.timeframe_weights = {
            "5m": 0.10,
            "15m": 0.15,
            "1h": 0.25,
            "4h": 0.30,
            "1d": 0.20
        }
        
        self.indicator_weights = {
            "rsi": 0.08,
            "macd": 0.10,
            "ema_alignment": 0.12,
            "bb_position": 0.08,
            "adx_trend": 0.10,
            "stoch": 0.06,
            "volume": 0.10,
            "mfi": 0.06,
            "obv": 0.05,
            "cci": 0.05,
            "schaff": 0.05,
            "fisher": 0.05,
            "advanced": 0.10
        }
        
        self.signal_history: Dict[str, List[FusedSignal]] = {}
    
    def score_timeframe(self, indicators: TimeframeIndicators) -> TimeframeScore:
        trend_signals = []
        momentum_signals = []
        volatility_signals = []
        volume_signals = []
        key_signals = []
        
        if indicators.ema_9 > indicators.ema_21 > indicators.ema_50:
            trend_signals.append(1.0)
            key_signals.append(f"{indicators.timeframe}: EMAs aligned bullish")
        elif indicators.ema_9 < indicators.ema_21 < indicators.ema_50:
            trend_signals.append(-1.0)
            key_signals.append(f"{indicators.timeframe}: EMAs aligned bearish")
        elif indicators.ema_9 > indicators.ema_21:
            trend_signals.append(0.5)
        elif indicators.ema_9 < indicators.ema_21:
            trend_signals.append(-0.5)
        else:
            trend_signals.append(0.0)
        
        if indicators.current_price > indicators.sma_200:
            trend_signals.append(0.8)
        else:
            trend_signals.append(-0.8)
        
        if indicators.adx > 25:
            adx_strength = min((indicators.adx - 25) / 50, 1.0)
            if indicators.plus_di > indicators.minus_di:
                trend_signals.append(adx_strength)
                key_signals.append(f"{indicators.timeframe}: Strong ADX trend UP")
            else:
                trend_signals.append(-adx_strength)
                key_signals.append(f"{indicators.timeframe}: Strong ADX trend DOWN")
        
        if indicators.rsi < 30:
            momentum_signals.append(0.9)
            key_signals.append(f"{indicators.timeframe}: RSI oversold ({indicators.rsi:.1f})")
        elif indicators.rsi > 70:
            momentum_signals.append(-0.9)
            key_signals.append(f"{indicators.timeframe}: RSI overbought ({indicators.rsi:.1f})")
        elif indicators.rsi < 40:
            momentum_signals.append(0.3)
        elif indicators.rsi > 60:
            momentum_signals.append(-0.3)
        else:
            momentum_signals.append(0.0)
        
        if indicators.macd > indicators.macd_signal:
            macd_strength = min(abs(indicators.macd_histogram) * 10, 1.0)
            momentum_signals.append(macd_strength)
            if indicators.macd_histogram > 0 and indicators.macd < 0:
                key_signals.append(f"{indicators.timeframe}: MACD bullish crossover")
        else:
            macd_strength = min(abs(indicators.macd_histogram) * 10, 1.0)
            momentum_signals.append(-macd_strength)
        
        if indicators.stoch_k < 20 and indicators.stoch_k > indicators.stoch_d:
            momentum_signals.append(0.8)
            key_signals.append(f"{indicators.timeframe}: Stoch oversold reversal")
        elif indicators.stoch_k > 80 and indicators.stoch_k < indicators.stoch_d:
            momentum_signals.append(-0.8)
        elif indicators.stoch_k > indicators.stoch_d:
            momentum_signals.append(0.3)
        else:
            momentum_signals.append(-0.3)
        
        if indicators.cci < -100:
            momentum_signals.append(0.7)
        elif indicators.cci > 100:
            momentum_signals.append(-0.7)
        else:
            momentum_signals.append(indicators.cci / 200)
        
        if indicators.mfi < 20:
            momentum_signals.append(0.8)
            key_signals.append(f"{indicators.timeframe}: MFI oversold")
        elif indicators.mfi > 80:
            momentum_signals.append(-0.8)
        else:
            momentum_signals.append((50 - indicators.mfi) / 100)
        
        if indicators.schaff_tc is not None:
            if indicators.schaff_tc < 25:
                momentum_signals.append(0.7)
            elif indicators.schaff_tc > 75:
                momentum_signals.append(-0.7)
            else:
                momentum_signals.append((50 - indicators.schaff_tc) / 100)
        
        if indicators.fisher is not None:
            momentum_signals.append(max(min(indicators.fisher, 1.0), -1.0))
        
        if indicators.cmo is not None:
            momentum_signals.append(indicators.cmo / 100)
        
        if indicators.pmo is not None:
            momentum_signals.append(max(min(indicators.pmo * 5, 1.0), -1.0))
        
        bb_position = (indicators.current_price - indicators.bb_lower) / (indicators.bb_upper - indicators.bb_lower) if indicators.bb_upper != indicators.bb_lower else 0.5
        
        if bb_position < 0.1:
            volatility_signals.append(0.9)
            key_signals.append(f"{indicators.timeframe}: Price at lower BB")
        elif bb_position > 0.9:
            volatility_signals.append(-0.9)
        else:
            volatility_signals.append((0.5 - bb_position) * 2)
        
        volatility_ratio = indicators.bb_width / indicators.current_price if indicators.current_price > 0 else 0.02
        if volatility_ratio < 0.02:
            volatility_signals.append(0.5)
            key_signals.append(f"{indicators.timeframe}: BB squeeze (breakout coming)")
        elif volatility_ratio > 0.08:
            volatility_signals.append(-0.3)
        
        if indicators.volume > indicators.volume_sma * 1.5:
            volume_signals.append(0.8)
            key_signals.append(f"{indicators.timeframe}: High volume")
        elif indicators.volume < indicators.volume_sma * 0.5:
            volume_signals.append(-0.5)
        else:
            volume_signals.append((indicators.volume / indicators.volume_sma - 1) if indicators.volume_sma > 0 else 0)
        
        volume_signals.append(max(min(indicators.obv_trend, 1.0), -1.0))
        
        trend_score = statistics.mean(trend_signals) if trend_signals else 0
        momentum_score = statistics.mean(momentum_signals) if momentum_signals else 0
        volatility_score = statistics.mean(volatility_signals) if volatility_signals else 0
        volume_score = statistics.mean(volume_signals) if volume_signals else 0
        
        overall = (
            trend_score * 0.30 +
            momentum_score * 0.35 +
            volatility_score * 0.15 +
            volume_score * 0.20
        )
        
        if overall > 0.3:
            direction = "BULLISH"
            reversal_prob = 0.0
            continuation_prob = min(overall + 0.3, 1.0)
        elif overall < -0.3:
            direction = "BEARISH"
            reversal_prob = 0.0
            continuation_prob = min(abs(overall) + 0.3, 1.0)
        else:
            direction = "NEUTRAL"
            reversal_prob = 0.5
            continuation_prob = 0.5
        
        if momentum_score < -0.5 and trend_score > 0:
            reversal_prob = 0.7
        elif momentum_score > 0.5 and trend_score < 0:
            reversal_prob = 0.7
        
        confidence = (abs(trend_score) + abs(momentum_score)) / 2
        confidence = min(confidence + 0.2, 0.95)
        
        return TimeframeScore(
            timeframe=indicators.timeframe,
            trend_score=trend_score,
            momentum_score=momentum_score,
            volatility_score=volatility_score,
            volume_score=volume_score,
            reversal_probability=reversal_prob,
            continuation_probability=continuation_prob,
            overall_score=overall,
            signal_direction=direction,
            confidence=confidence,
            key_signals=key_signals
        )
    
    def calculate_consensus(self, scores: Dict[str, TimeframeScore]) -> MTFConsensus:
        bullish = 0
        bearish = 0
        neutral = 0
        divergences = []
        
        for tf, score in scores.items():
            if score.signal_direction == "BULLISH":
                bullish += 1
            elif score.signal_direction == "BEARISH":
                bearish += 1
            else:
                neutral += 1
        
        short_tfs = ["5m", "15m"]
        long_tfs = ["4h", "1d"]
        
        short_bullish = sum(1 for tf in short_tfs if tf in scores and scores[tf].signal_direction == "BULLISH")
        long_bullish = sum(1 for tf in long_tfs if tf in scores and scores[tf].signal_direction == "BULLISH")
        short_bearish = sum(1 for tf in short_tfs if tf in scores and scores[tf].signal_direction == "BEARISH")
        long_bearish = sum(1 for tf in long_tfs if tf in scores and scores[tf].signal_direction == "BEARISH")
        
        if short_bearish >= 2 and long_bullish >= 1:
            divergences.append("SHORT-TERM BEARISH / LONG-TERM BULLISH (potential dip buy)")
        if short_bullish >= 2 and long_bearish >= 1:
            divergences.append("SHORT-TERM BULLISH / LONG-TERM BEARISH (potential bull trap)")
        
        if "1h" in scores and "4h" in scores:
            if scores["1h"].momentum_score > 0.5 and scores["4h"].momentum_score < -0.3:
                divergences.append("1H/4H momentum divergence - caution")
        
        if bullish > bearish and bullish > neutral:
            direction = "BULLISH"
            strength = bullish / (bullish + bearish + neutral)
        elif bearish > bullish and bearish > neutral:
            direction = "BEARISH"
            strength = bearish / (bullish + bearish + neutral)
        else:
            direction = "NEUTRAL"
            strength = 0.5
        
        all_directions = [s.signal_direction for s in scores.values()]
        if len(set(all_directions)) == 1:
            alignment = 1.0
        elif len(set(all_directions)) == 2 and "NEUTRAL" in all_directions:
            alignment = 0.7
        else:
            alignment = 0.3
        
        return MTFConsensus(
            bullish_count=bullish,
            bearish_count=bearish,
            neutral_count=neutral,
            consensus_direction=direction,
            consensus_strength=strength,
            divergences=divergences,
            alignment_score=alignment
        )
    
    def fuse_signals(
        self,
        symbol: str,
        timeframe_indicators: Dict[str, TimeframeIndicators],
        kpi_data: Optional[Dict[str, Any]] = None
    ) -> FusedSignal:
        scores: Dict[str, TimeframeScore] = {}
        all_key_signals = []
        
        for tf, indicators in timeframe_indicators.items():
            score = self.score_timeframe(indicators)
            scores[tf] = score
            all_key_signals.extend(score.key_signals)
        
        consensus = self.calculate_consensus(scores)
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for tf, score in scores.items():
            weight = self.timeframe_weights.get(tf, 0.1)
            weighted_score += score.overall_score * weight
            total_weight += weight
        
        if total_weight > 0:
            weighted_score /= total_weight
        
        if kpi_data:
            kpi_boost = 0.0
            if kpi_data.get("win_rate", 0) > 0.6:
                kpi_boost += 0.05
            if kpi_data.get("sharpe_ratio", 0) > 1.5:
                kpi_boost += 0.05
            if kpi_data.get("profit_factor", 0) > 2.0:
                kpi_boost += 0.05
            weighted_score += kpi_boost
        
        if consensus.alignment_score > 0.8:
            strength = min(abs(weighted_score) * 1.3, 1.0)
        else:
            strength = min(abs(weighted_score) * 0.9, 0.85)
        
        confidence = (
            consensus.alignment_score * 0.4 +
            consensus.consensus_strength * 0.3 +
            statistics.mean([s.confidence for s in scores.values()]) * 0.3
        )
        
        if weighted_score > 0.2:
            direction = "LONG"
        elif weighted_score < -0.2:
            direction = "SHORT"
        else:
            direction = "NEUTRAL"
        
        warnings = []
        if consensus.divergences:
            warnings.extend(consensus.divergences)
        if consensus.alignment_score < 0.5:
            warnings.append("Low timeframe alignment - conflicting signals")
        
        entry_score = strength * confidence
        
        reversal_probs = [s.reversal_probability for s in scores.values()]
        timing_score = 1.0 - statistics.mean(reversal_probs)
        
        risk_score = 1.0 - confidence
        if len(warnings) > 0:
            risk_score += 0.1 * len(warnings)
        risk_score = min(risk_score, 1.0)
        
        if entry_score > 0.7 and risk_score < 0.4:
            action = "STRONG_BUY" if direction == "LONG" else "STRONG_SELL"
        elif entry_score > 0.5 and risk_score < 0.5:
            action = "BUY" if direction == "LONG" else "SELL"
        elif entry_score > 0.3:
            action = "WEAK_BUY" if direction == "LONG" else "WEAK_SELL"
        else:
            action = "HOLD"
        
        current_price = list(timeframe_indicators.values())[0].current_price
        atr = list(timeframe_indicators.values())[0].atr
        
        if direction == "LONG":
            entry_zone = (current_price * 0.995, current_price * 1.005)
            stop_zone = (current_price - atr * 2, current_price - atr * 1.5)
            targets = [
                (current_price + atr * 1.5, current_price + atr * 2),
                (current_price + atr * 3, current_price + atr * 4),
                (current_price + atr * 5, current_price + atr * 6)
            ]
        else:
            entry_zone = (current_price * 0.995, current_price * 1.005)
            stop_zone = (current_price + atr * 1.5, current_price + atr * 2)
            targets = [
                (current_price - atr * 2, current_price - atr * 1.5),
                (current_price - atr * 4, current_price - atr * 3),
                (current_price - atr * 6, current_price - atr * 5)
            ]
        
        final_score = entry_score * 100
        if final_score >= 85:
            grade = "A+"
        elif final_score >= 75:
            grade = "A"
        elif final_score >= 65:
            grade = "B+"
        elif final_score >= 55:
            grade = "B"
        elif final_score >= 45:
            grade = "C+"
        elif final_score >= 35:
            grade = "C"
        else:
            grade = "D"
        
        beats_competition = (
            confidence * 0.3 +
            consensus.alignment_score * 0.3 +
            (len(timeframe_indicators) / 5) * 0.2 +
            (len(all_key_signals) / 20) * 0.2
        ) * 100
        
        fused = FusedSignal(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            direction=direction,
            strength=strength,
            confidence=confidence,
            entry_score=entry_score,
            timing_score=timing_score,
            risk_score=risk_score,
            timeframe_scores=scores,
            consensus=consensus,
            primary_drivers=all_key_signals[:10],
            warnings=warnings,
            recommended_action=action,
            optimal_entry_zone=entry_zone,
            stop_loss_zone=stop_zone,
            target_zones=targets,
            overall_grade=grade,
            beats_competition_score=beats_competition
        )
        
        if symbol not in self.signal_history:
            self.signal_history[symbol] = []
        self.signal_history[symbol].append(fused)
        self.signal_history[symbol] = self.signal_history[symbol][-50:]
        
        return fused
    
    def get_ultra_signal(
        self,
        symbol: str,
        timeframe_indicators: Dict[str, TimeframeIndicators],
        kpi_data: Optional[Dict[str, Any]] = None,
        ml_confidence: float = 0.5,
        whale_activity: bool = False,
        smart_money_flow: float = 0.0
    ) -> FusedSignal:
        base_signal = self.fuse_signals(symbol, timeframe_indicators, kpi_data)
        
        if ml_confidence > 0.7:
            base_signal.confidence = min(base_signal.confidence + 0.1, 0.98)
            base_signal.entry_score = min(base_signal.entry_score + 0.1, 1.0)
            base_signal.primary_drivers.append(f"ML confidence high: {ml_confidence:.1%}")
        
        if whale_activity and base_signal.direction == "LONG":
            base_signal.confidence = min(base_signal.confidence + 0.08, 0.98)
            base_signal.primary_drivers.append("Whale accumulation detected")
        
        if smart_money_flow > 0.5:
            base_signal.confidence = min(base_signal.confidence + 0.05, 0.98)
            base_signal.primary_drivers.append(f"Smart money inflow: {smart_money_flow:.1%}")
        elif smart_money_flow < -0.5:
            base_signal.warnings.append(f"Smart money outflow: {smart_money_flow:.1%}")
            base_signal.risk_score = min(base_signal.risk_score + 0.1, 1.0)
        
        base_signal.beats_competition_score = min(base_signal.beats_competition_score + 15, 100)
        
        return base_signal
