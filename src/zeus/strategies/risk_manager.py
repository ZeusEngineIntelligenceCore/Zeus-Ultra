#!/usr/bin/env python3
"""
ZEUS RISK MANAGER - Professional Portfolio Risk Management
Implements Kelly Criterion, position sizing, and drawdown controls
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
from enum import Enum

from ..exchanges.base import Position, OrderSide


class RiskLevel(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class RiskConfig:
    max_risk_per_trade: float = 0.50
    max_portfolio_risk: float = 0.90
    max_open_positions: int = 100
    max_position_size_pct: float = 0.90
    max_daily_loss_pct: float = 1.0
    max_drawdown_pct: float = 1.0
    max_realized_loss_usd: float = 20.0
    kelly_fraction: float = 0.25
    min_win_rate: float = 0.40
    trailing_stop_activation: float = 0.015
    trailing_stop_distance: float = 0.01
    break_even_activation: float = 0.01
    partial_tp_levels: List[float] = field(default_factory=lambda: [0.33, 0.33, 0.34])
    correlation_limit: float = 0.7
    risk_level: RiskLevel = RiskLevel.MODERATE


@dataclass
class TradeStats:
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit: float = 0.0
    total_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    current_streak: int = 0
    daily_pnl: float = 0.0
    daily_trades: int = 0
    last_trade_time: Optional[datetime] = None

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def profit_factor(self) -> float:
        if self.total_loss == 0:
            return float('inf') if self.total_profit > 0 else 0.0
        return abs(self.total_profit / self.total_loss)

    @property
    def avg_win(self) -> float:
        if self.winning_trades == 0:
            return 0.0
        return self.total_profit / self.winning_trades

    @property
    def avg_loss(self) -> float:
        if self.losing_trades == 0:
            return 0.0
        return abs(self.total_loss / self.losing_trades)

    @property
    def expectancy(self) -> float:
        return (self.win_rate * self.avg_win) - ((1 - self.win_rate) * self.avg_loss)


@dataclass
class PortfolioState:
    equity: float = 0.0
    peak_equity: float = 0.0
    available_balance: float = 0.0
    total_exposure: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    positions: Dict[str, Position] = field(default_factory=dict)
    stats: TradeStats = field(default_factory=TradeStats)

    @property
    def drawdown(self) -> float:
        if self.peak_equity == 0:
            return 0.0
        return (self.peak_equity - self.equity) / self.peak_equity

    @property
    def exposure_pct(self) -> float:
        if self.equity == 0:
            return 0.0
        return self.total_exposure / self.equity

    @property
    def position_count(self) -> int:
        return len([p for p in self.positions.values() if p.status == "open"])


class RiskManager:
    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        self.portfolio = PortfolioState()
        self._daily_reset_time: Optional[datetime] = None

    def update_portfolio(self, equity: float, positions: List[Position]) -> None:
        self.portfolio.equity = equity
        self.portfolio.peak_equity = max(self.portfolio.peak_equity, equity)
        self.portfolio.positions = {p.symbol: p for p in positions if p.status == "open"}
        self.portfolio.total_exposure = sum(
            abs(p.size * p.current_price) for p in positions if p.status == "open"
        )
        self.portfolio.unrealized_pnl = sum(
            p.unrealized_pnl for p in positions if p.status == "open"
        )
        self._check_daily_reset()

    def _check_daily_reset(self) -> None:
        now = datetime.now(timezone.utc)
        if self._daily_reset_time is None or now.date() > self._daily_reset_time.date():
            self.portfolio.stats.daily_pnl = 0.0
            self.portfolio.stats.daily_trades = 0
            self._daily_reset_time = now

    def can_open_position(self, symbol: str) -> tuple:
        reasons = []
        if self.portfolio.position_count >= self.config.max_open_positions:
            reasons.append(f"Max positions reached ({self.config.max_open_positions})")
        realized_loss = abs(min(0, self.portfolio.stats.total_loss))
        if realized_loss >= self.config.max_realized_loss_usd:
            reasons.append(f"Max realized loss reached (${realized_loss:.2f} of ${self.config.max_realized_loss_usd:.2f})")
        if self.portfolio.exposure_pct >= self.config.max_portfolio_risk * 10:
            reasons.append(f"Portfolio exposure too high ({self.portfolio.exposure_pct:.1%})")
        if symbol in self.portfolio.positions:
            reasons.append(f"Already have position in {symbol}")
        return len(reasons) == 0, reasons

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        confidence: float = 100.0
    ) -> float:
        if entry_price <= 0 or stop_loss <= 0:
            return 0.0
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit <= 0:
            return 0.0
        base_risk = self.config.max_risk_per_trade
        if self.config.risk_level == RiskLevel.CONSERVATIVE:
            base_risk *= 0.5
        elif self.config.risk_level == RiskLevel.AGGRESSIVE:
            base_risk *= 1.5
        confidence_factor = min(1.0, confidence / 100.0)
        adjusted_risk = base_risk * (0.5 + 0.5 * confidence_factor)
        if self.portfolio.stats.consecutive_losses >= 3:
            adjusted_risk *= 0.5
        if self.portfolio.drawdown > 0.05:
            drawdown_factor = 1 - (self.portfolio.drawdown / self.config.max_drawdown_pct)
            adjusted_risk *= max(0.25, drawdown_factor)
        risk_amount = self.portfolio.equity * adjusted_risk
        position_size = risk_amount / risk_per_unit
        max_size_by_pct = (self.portfolio.equity * self.config.max_position_size_pct) / entry_price
        position_size = min(position_size, max_size_by_pct)
        remaining_exposure = (self.portfolio.equity * self.config.max_portfolio_risk * 10) - self.portfolio.total_exposure
        max_size_by_exposure = remaining_exposure / entry_price
        position_size = min(position_size, max(0, max_size_by_exposure))
        return max(0, position_size)

    def calculate_kelly_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        if avg_loss <= 0 or win_rate <= 0:
            return 0.0
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        kelly = (b * p - q) / b
        kelly = max(0, kelly * self.config.kelly_fraction)
        return min(kelly, self.config.max_risk_per_trade * 2)

    def calculate_optimal_size(
        self,
        entry_price: float,
        stop_loss: float,
        confidence: float = 100.0
    ) -> float:
        standard_size = self.calculate_position_size(entry_price, stop_loss, confidence)
        if self.portfolio.stats.total_trades >= 20 and self.portfolio.stats.win_rate >= self.config.min_win_rate:
            kelly_risk = self.calculate_kelly_size(
                self.portfolio.stats.win_rate,
                self.portfolio.stats.avg_win,
                self.portfolio.stats.avg_loss
            )
            if kelly_risk > 0:
                risk_per_unit = abs(entry_price - stop_loss)
                kelly_size = (self.portfolio.equity * kelly_risk) / risk_per_unit
                return min(standard_size, kelly_size)
        return standard_size

    def calculate_trailing_stop(
        self,
        position: Position,
        current_price: float
    ) -> Optional[float]:
        if position.stop_loss is None:
            return None
        entry = position.entry_price
        pnl_pct = (current_price - entry) / entry if position.side == OrderSide.BUY else (entry - current_price) / entry
        if pnl_pct >= self.config.break_even_activation:
            break_even = entry * 1.001 if position.side == OrderSide.BUY else entry * 0.999
            if position.side == OrderSide.BUY:
                new_stop = max(position.stop_loss, break_even)
            else:
                new_stop = min(position.stop_loss, break_even)
            if pnl_pct >= self.config.trailing_stop_activation:
                trailing_distance = current_price * self.config.trailing_stop_distance
                if position.side == OrderSide.BUY:
                    trail_stop = current_price - trailing_distance
                    new_stop = max(new_stop, trail_stop)
                else:
                    trail_stop = current_price + trailing_distance
                    new_stop = min(new_stop, trail_stop)
            return new_stop
        return None

    def should_close_position(self, position: Position, current_price: float) -> tuple:
        reasons = []
        if position.stop_loss and current_price <= position.stop_loss and position.side == OrderSide.BUY:
            reasons.append("Stop loss hit")
        if position.stop_loss and current_price >= position.stop_loss and position.side == OrderSide.SELL:
            reasons.append("Stop loss hit")
        if position.take_profit and current_price >= position.take_profit and position.side == OrderSide.BUY:
            reasons.append("Take profit hit")
        if position.take_profit and current_price <= position.take_profit and position.side == OrderSide.SELL:
            reasons.append("Take profit hit")
        position_pnl_pct = position.unrealized_pnl / (position.entry_price * position.size) if position.size > 0 else 0
        if position_pnl_pct <= -0.05:
            reasons.append("Emergency stop: -5% loss")
        return len(reasons) > 0, reasons

    def record_trade(self, pnl: float, is_win: bool) -> None:
        self.portfolio.stats.total_trades += 1
        self.portfolio.stats.daily_trades += 1
        self.portfolio.stats.daily_pnl += pnl
        self.portfolio.stats.last_trade_time = datetime.now(timezone.utc)
        if is_win:
            self.portfolio.stats.winning_trades += 1
            self.portfolio.stats.total_profit += pnl
            self.portfolio.stats.largest_win = max(self.portfolio.stats.largest_win, pnl)
            self.portfolio.stats.consecutive_wins += 1
            self.portfolio.stats.consecutive_losses = 0
            self.portfolio.stats.max_consecutive_wins = max(
                self.portfolio.stats.max_consecutive_wins,
                self.portfolio.stats.consecutive_wins
            )
            self.portfolio.stats.current_streak = self.portfolio.stats.consecutive_wins
        else:
            self.portfolio.stats.losing_trades += 1
            self.portfolio.stats.total_loss += pnl
            self.portfolio.stats.largest_loss = min(self.portfolio.stats.largest_loss, pnl)
            self.portfolio.stats.consecutive_losses += 1
            self.portfolio.stats.consecutive_wins = 0
            self.portfolio.stats.max_consecutive_losses = max(
                self.portfolio.stats.max_consecutive_losses,
                self.portfolio.stats.consecutive_losses
            )
            self.portfolio.stats.current_streak = -self.portfolio.stats.consecutive_losses

    def get_risk_report(self) -> Dict[str, Any]:
        stats = self.portfolio.stats
        return {
            "equity": round(self.portfolio.equity, 2),
            "peak_equity": round(self.portfolio.peak_equity, 2),
            "drawdown_pct": round(self.portfolio.drawdown * 100, 2),
            "exposure_pct": round(self.portfolio.exposure_pct * 100, 2),
            "open_positions": self.portfolio.position_count,
            "max_positions": self.config.max_open_positions,
            "unrealized_pnl": round(self.portfolio.unrealized_pnl, 2),
            "daily_pnl": round(stats.daily_pnl, 2),
            "daily_trades": stats.daily_trades,
            "total_trades": stats.total_trades,
            "win_rate": round(stats.win_rate * 100, 2),
            "profit_factor": round(stats.profit_factor, 2),
            "avg_win": round(stats.avg_win, 2),
            "avg_loss": round(stats.avg_loss, 2),
            "expectancy": round(stats.expectancy, 2),
            "current_streak": stats.current_streak,
            "consecutive_losses": stats.consecutive_losses,
            "max_consecutive_losses": stats.max_consecutive_losses,
            "risk_level": self.config.risk_level.value
        }
