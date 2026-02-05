#!/usr/bin/env python3
"""
ZEUS ADVANCED EXIT STRATEGIES
Quick wins, trailing stops, partial profit taking, and smart exit timing
"""

from __future__ import annotations
import math
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger("Zeus.ExitStrategies")


class ExitReason(Enum):
    QUICK_WIN = "quick_win"
    PARTIAL_PROFIT = "partial_profit"
    TRAILING_STOP = "trailing_stop"
    MOMENTUM_REVERSAL = "momentum_reversal"
    VOLUME_COLLAPSE = "volume_collapse"
    FAKEOUT_DETECTED = "fakeout_detected"
    TIME_BASED = "time_based"
    TARGET_HIT = "target_hit"
    STOP_LOSS = "stop_loss"


@dataclass
class ExitSignal:
    should_exit: bool
    reason: ExitReason
    exit_size_pct: float  # 0-100, percentage of position to exit
    urgency: str  # "immediate", "normal", "low"
    confidence: float  # 0-1
    suggested_price: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExitConfig:
    quick_win_threshold: float = 0.015  # 1.5% quick profit
    quick_win_time_limit_mins: int = 15  # Within first 15 mins
    partial_profit_levels: List[float] = field(default_factory=lambda: [0.02, 0.04, 0.06])  # 2%, 4%, 6%
    partial_profit_sizes: List[float] = field(default_factory=lambda: [0.33, 0.33, 0.34])  # Exit percentages
    trailing_stop_activation: float = 0.02  # Activate after 2% profit
    trailing_stop_distance: float = 0.008  # 0.8% trailing distance
    momentum_reversal_threshold: float = -0.3  # RSI/MACD reversal strength
    volume_collapse_threshold: float = 0.3  # Volume drops to 30% of avg
    time_decay_start_mins: int = 60  # Start penalizing after 1 hour
    max_hold_time_mins: int = 480  # 8 hours max hold


class AdvancedExitManager:
    def __init__(self, config: Optional[ExitConfig] = None):
        self.config = config or ExitConfig()
        self.partial_exits_made: Dict[str, List[int]] = {}  # trade_id -> levels exited
    
    def check_quick_win(
        self,
        trade_id: str,
        entry_price: float,
        current_price: float,
        entry_time: datetime,
        atr: float
    ) -> Optional[ExitSignal]:
        now = datetime.now(timezone.utc)
        hold_time_mins = (now - entry_time).total_seconds() / 60
        
        if hold_time_mins > self.config.quick_win_time_limit_mins:
            return None
        
        profit_pct = (current_price - entry_price) / entry_price
        
        dynamic_threshold = self.config.quick_win_threshold
        if atr > 0:
            atr_pct = atr / entry_price
            if atr_pct > 0.02:  # High volatility
                dynamic_threshold *= 1.5
            elif atr_pct < 0.01:  # Low volatility
                dynamic_threshold *= 0.7
        
        if profit_pct >= dynamic_threshold:
            time_bonus = 1.0 - (hold_time_mins / self.config.quick_win_time_limit_mins) * 0.3
            confidence = min(1.0, profit_pct / dynamic_threshold * time_bonus)
            
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.QUICK_WIN,
                exit_size_pct=100.0,
                urgency="immediate",
                confidence=confidence,
                suggested_price=current_price,
                details={
                    "profit_pct": round(profit_pct * 100, 2),
                    "hold_time_mins": round(hold_time_mins, 1),
                    "threshold_used": round(dynamic_threshold * 100, 2)
                }
            )
        
        return None
    
    def check_partial_profits(
        self,
        trade_id: str,
        entry_price: float,
        current_price: float,
        remaining_size_pct: float = 100.0
    ) -> Optional[ExitSignal]:
        if trade_id not in self.partial_exits_made:
            self.partial_exits_made[trade_id] = []
        
        profit_pct = (current_price - entry_price) / entry_price
        levels_taken = self.partial_exits_made[trade_id]
        
        for i, level in enumerate(self.config.partial_profit_levels):
            if i in levels_taken:
                continue
            
            if profit_pct >= level:
                exit_size = self.config.partial_profit_sizes[i] * remaining_size_pct
                self.partial_exits_made[trade_id].append(i)
                
                return ExitSignal(
                    should_exit=True,
                    reason=ExitReason.PARTIAL_PROFIT,
                    exit_size_pct=exit_size,
                    urgency="normal",
                    confidence=0.8,
                    suggested_price=current_price,
                    details={
                        "profit_pct": round(profit_pct * 100, 2),
                        "level": i + 1,
                        "total_levels": len(self.config.partial_profit_levels)
                    }
                )
        
        return None
    
    def calculate_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        highest_price: float,
        atr: float
    ) -> Tuple[bool, float, float]:
        profit_pct = (current_price - entry_price) / entry_price
        
        if profit_pct < self.config.trailing_stop_activation:
            return False, 0.0, 0.0
        
        atr_pct = atr / entry_price if entry_price > 0 else 0.02
        dynamic_distance = max(
            self.config.trailing_stop_distance,
            min(atr_pct * 0.5, 0.02)  # 50% of ATR, max 2%
        )
        
        profit_from_high = (highest_price - entry_price) / entry_price
        if profit_from_high > 0.05:  # >5% from entry
            dynamic_distance *= 0.8  # Tighter stop
        
        trailing_stop_price = highest_price * (1 - dynamic_distance)
        
        if current_price <= trailing_stop_price:
            return True, trailing_stop_price, dynamic_distance
        
        return False, trailing_stop_price, dynamic_distance
    
    def check_momentum_reversal(
        self,
        rsi_current: float,
        rsi_prev: float,
        macd_histogram: float,
        macd_prev_histogram: float,
        profit_pct: float
    ) -> Optional[ExitSignal]:
        if profit_pct <= 0:
            return None
        
        rsi_reversal = False
        macd_reversal = False
        reversal_strength = 0.0
        
        if rsi_prev >= 70 and rsi_current < 70:
            rsi_reversal = True
            reversal_strength += (70 - rsi_current) / 20
        
        if rsi_prev >= 80 and rsi_current < 75:
            reversal_strength += 0.3
        
        if macd_prev_histogram > 0 and macd_histogram < 0:
            macd_reversal = True
            reversal_strength += abs(macd_histogram) / (abs(macd_prev_histogram) + 1e-10) * 0.5
        
        if macd_prev_histogram > macd_histogram and macd_histogram < macd_prev_histogram * 0.5:
            reversal_strength += 0.2
        
        if reversal_strength > abs(self.config.momentum_reversal_threshold):
            urgency = "immediate" if reversal_strength > 0.6 else "normal"
            
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.MOMENTUM_REVERSAL,
                exit_size_pct=min(100, 50 + reversal_strength * 50),
                urgency=urgency,
                confidence=min(1.0, reversal_strength),
                details={
                    "rsi_reversal": rsi_reversal,
                    "macd_reversal": macd_reversal,
                    "reversal_strength": round(reversal_strength, 2),
                    "rsi": round(rsi_current, 1),
                    "profit_at_signal": round(profit_pct * 100, 2)
                }
            )
        
        return None
    
    def check_volume_collapse(
        self,
        current_volume: float,
        avg_volume: float,
        profit_pct: float
    ) -> Optional[ExitSignal]:
        if profit_pct <= 0:
            return None
        
        volume_ratio = current_volume / (avg_volume + 1e-10)
        
        if volume_ratio < self.config.volume_collapse_threshold:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.VOLUME_COLLAPSE,
                exit_size_pct=50.0,
                urgency="normal",
                confidence=1.0 - volume_ratio,
                details={
                    "volume_ratio": round(volume_ratio, 2),
                    "current_volume": current_volume,
                    "avg_volume": round(avg_volume, 2)
                }
            )
        
        return None
    
    def check_time_based_exit(
        self,
        entry_time: datetime,
        profit_pct: float,
        volatility: float
    ) -> Optional[ExitSignal]:
        now = datetime.now(timezone.utc)
        hold_time_mins = (now - entry_time).total_seconds() / 60
        
        max_hold = self.config.max_hold_time_mins
        if volatility < 0.01:
            max_hold *= 1.5
        elif volatility > 0.03:
            max_hold *= 0.7
        
        if hold_time_mins > max_hold:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.TIME_BASED,
                exit_size_pct=100.0,
                urgency="low" if profit_pct > 0 else "normal",
                confidence=0.7,
                details={
                    "hold_time_mins": round(hold_time_mins, 1),
                    "max_hold_mins": round(max_hold, 1),
                    "profit_pct": round(profit_pct * 100, 2)
                }
            )
        
        if hold_time_mins > self.config.time_decay_start_mins and profit_pct > 0:
            decay_factor = (hold_time_mins - self.config.time_decay_start_mins) / (max_hold - self.config.time_decay_start_mins)
            if decay_factor > 0.5 and profit_pct > 0.01:
                return ExitSignal(
                    should_exit=True,
                    reason=ExitReason.TIME_BASED,
                    exit_size_pct=50.0,
                    urgency="low",
                    confidence=decay_factor * 0.6,
                    details={
                        "hold_time_mins": round(hold_time_mins, 1),
                        "decay_factor": round(decay_factor, 2),
                        "partial_exit": True
                    }
                )
        
        return None
    
    def get_optimal_exit(
        self,
        trade_id: str,
        entry_price: float,
        current_price: float,
        highest_price: float,
        entry_time: datetime,
        atr: float,
        rsi: float,
        rsi_prev: float,
        macd_histogram: float,
        macd_prev_histogram: float,
        current_volume: float,
        avg_volume: float,
        remaining_size_pct: float = 100.0
    ) -> Optional[ExitSignal]:
        profit_pct = (current_price - entry_price) / entry_price
        volatility = atr / entry_price if entry_price > 0 else 0.02
        
        quick_win = self.check_quick_win(trade_id, entry_price, current_price, entry_time, atr)
        if quick_win:
            return quick_win
        
        trailing_hit, stop_price, distance = self.calculate_trailing_stop(
            entry_price, current_price, highest_price, atr
        )
        if trailing_hit:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.TRAILING_STOP,
                exit_size_pct=100.0,
                urgency="immediate",
                confidence=0.9,
                suggested_price=stop_price,
                details={
                    "stop_price": round(stop_price, 8),
                    "distance_pct": round(distance * 100, 2),
                    "highest_price": round(highest_price, 8)
                }
            )
        
        momentum = self.check_momentum_reversal(
            rsi, rsi_prev, macd_histogram, macd_prev_histogram, profit_pct
        )
        if momentum and momentum.confidence > 0.5:
            return momentum
        
        volume_signal = self.check_volume_collapse(current_volume, avg_volume, profit_pct)
        if volume_signal and volume_signal.confidence > 0.6:
            return volume_signal
        
        partial = self.check_partial_profits(trade_id, entry_price, current_price, remaining_size_pct)
        if partial:
            return partial
        
        time_exit = self.check_time_based_exit(entry_time, profit_pct, volatility)
        if time_exit:
            return time_exit
        
        return None
    
    def cleanup_trade(self, trade_id: str) -> None:
        if trade_id in self.partial_exits_made:
            del self.partial_exits_made[trade_id]
