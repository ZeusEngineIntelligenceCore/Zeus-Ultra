#!/usr/bin/env python3
"""
ZEUS KPI TRACKER - Advanced Performance Analytics
Tracks 10 critical trading KPIs for comprehensive performance monitoring
"""

from __future__ import annotations
import math
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path

logger = logging.getLogger("Zeus.KPI")


@dataclass
class TradeMetrics:
    symbol: str
    pnl: float
    pnl_pct: float
    hold_time_seconds: int
    entry_price: float
    exit_price: float
    size: float
    confidence: float
    timestamp: str


@dataclass
class KPIState:
    trades: List[TradeMetrics] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)
    peak_equity: float = 0.0
    current_equity: float = 0.0
    starting_equity: float = 0.0
    last_updated: str = ""


class KPITracker:
    def __init__(self, state_file: str = "data/kpi_state.json"):
        self.state_file = Path(state_file)
        self.state = KPIState()
        self._load_state()
    
    def _load_state(self) -> None:
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.state.daily_returns = data.get("daily_returns", [])
                    self.state.peak_equity = data.get("peak_equity", 0.0)
                    self.state.current_equity = data.get("current_equity", 0.0)
                    self.state.starting_equity = data.get("starting_equity", 0.0)
                    for t in data.get("trades", []):
                        self.state.trades.append(TradeMetrics(**t))
                logger.info(f"Loaded KPI state with {len(self.state.trades)} trades")
            except Exception as e:
                logger.error(f"Failed to load KPI state: {e}")
    
    def _save_state(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = {
                "trades": [
                    {
                        "symbol": t.symbol,
                        "pnl": t.pnl,
                        "pnl_pct": t.pnl_pct,
                        "hold_time_seconds": t.hold_time_seconds,
                        "entry_price": t.entry_price,
                        "exit_price": t.exit_price,
                        "size": t.size,
                        "confidence": t.confidence,
                        "timestamp": t.timestamp
                    }
                    for t in self.state.trades[-500:]
                ],
                "daily_returns": self.state.daily_returns[-365:],
                "peak_equity": self.state.peak_equity,
                "current_equity": self.state.current_equity,
                "starting_equity": self.state.starting_equity,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save KPI state: {e}")
    
    def record_trade(self, trade: TradeMetrics) -> None:
        self.state.trades.append(trade)
        self._save_state()
    
    def update_equity(self, equity: float) -> None:
        if self.state.starting_equity == 0:
            self.state.starting_equity = equity
        self.state.current_equity = equity
        self.state.peak_equity = max(self.state.peak_equity, equity)
        self._save_state()
    
    def record_daily_return(self, return_pct: float) -> None:
        self.state.daily_returns.append(return_pct)
        self._save_state()

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        if len(self.state.daily_returns) < 30:
            return 0.0
        returns = self.state.daily_returns[-252:]
        avg_return = sum(returns) / len(returns)
        std_dev = math.sqrt(sum((r - avg_return) ** 2 for r in returns) / len(returns))
        if std_dev < 1e-9:
            return 0.0
        daily_rf = risk_free_rate / 252
        sharpe = (avg_return - daily_rf) / std_dev * math.sqrt(252)
        return round(sharpe, 3)

    def calculate_sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        if len(self.state.daily_returns) < 30:
            return 0.0
        returns = self.state.daily_returns[-252:]
        avg_return = sum(returns) / len(returns)
        downside_returns = [r for r in returns if r < 0]
        if len(downside_returns) < 5:
            return 0.0
        downside_std = math.sqrt(sum(r ** 2 for r in downside_returns) / len(downside_returns))
        if downside_std < 1e-9:
            return 0.0
        daily_rf = risk_free_rate / 252
        sortino = (avg_return - daily_rf) / downside_std * math.sqrt(252)
        return round(sortino, 3)

    def calculate_max_drawdown(self) -> Tuple[float, float]:
        if self.state.peak_equity == 0:
            return 0.0, 0.0
        current_dd = (self.state.peak_equity - self.state.current_equity) / self.state.peak_equity * 100
        max_dd = 0.0
        peak = 0.0
        for ret in self.state.daily_returns:
            equity = peak * (1 + ret / 100) if peak > 0 else 100 * (1 + ret / 100)
            peak = max(peak, equity)
            dd = (peak - equity) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)
        return round(max_dd, 2), round(current_dd, 2)

    def calculate_win_rate(self) -> float:
        if not self.state.trades:
            return 0.0
        wins = sum(1 for t in self.state.trades if t.pnl > 0)
        return round(wins / len(self.state.trades) * 100, 2)

    def calculate_profit_factor(self) -> float:
        gross_profit = sum(t.pnl for t in self.state.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.state.trades if t.pnl < 0))
        if gross_loss < 1e-9:
            return float('inf') if gross_profit > 0 else 0.0
        return round(gross_profit / gross_loss, 3)

    def calculate_average_win_loss(self) -> Tuple[float, float, float]:
        wins = [t.pnl for t in self.state.trades if t.pnl > 0]
        losses = [t.pnl for t in self.state.trades if t.pnl < 0]
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
        ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
        return round(avg_win, 2), round(avg_loss, 2), round(ratio, 3)

    def calculate_expectancy(self) -> float:
        if not self.state.trades:
            return 0.0
        win_rate = self.calculate_win_rate() / 100
        avg_win, avg_loss, _ = self.calculate_average_win_loss()
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        return round(expectancy, 2)

    def calculate_calmar_ratio(self) -> float:
        if len(self.state.daily_returns) < 30:
            return 0.0
        max_dd, _ = self.calculate_max_drawdown()
        if max_dd < 1e-9:
            return 0.0
        total_return = sum(self.state.daily_returns[-365:])
        calmar = total_return / max_dd
        return round(calmar, 3)

    def calculate_trade_frequency(self) -> Dict[str, float]:
        if not self.state.trades:
            return {"daily": 0, "weekly": 0, "monthly": 0}
        now = datetime.now(timezone.utc)
        day_ago = now - timedelta(days=1)
        week_ago = now - timedelta(days=7)
        month_ago = now - timedelta(days=30)
        daily = sum(1 for t in self.state.trades if datetime.fromisoformat(t.timestamp.replace('Z', '+00:00')) > day_ago)
        weekly = sum(1 for t in self.state.trades if datetime.fromisoformat(t.timestamp.replace('Z', '+00:00')) > week_ago)
        monthly = sum(1 for t in self.state.trades if datetime.fromisoformat(t.timestamp.replace('Z', '+00:00')) > month_ago)
        return {"daily": daily, "weekly": weekly, "monthly": monthly}

    def calculate_avg_hold_time(self) -> float:
        if not self.state.trades:
            return 0.0
        total_time = sum(t.hold_time_seconds for t in self.state.trades)
        return round(total_time / len(self.state.trades) / 3600, 2)

    def get_all_kpis(self) -> Dict[str, Any]:
        sharpe = self.calculate_sharpe_ratio()
        sortino = self.calculate_sortino_ratio()
        max_dd, current_dd = self.calculate_max_drawdown()
        win_rate = self.calculate_win_rate()
        profit_factor = self.calculate_profit_factor()
        avg_win, avg_loss, win_loss_ratio = self.calculate_average_win_loss()
        expectancy = self.calculate_expectancy()
        calmar = self.calculate_calmar_ratio()
        frequency = self.calculate_trade_frequency()
        avg_hold = self.calculate_avg_hold_time()
        
        return {
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown_pct": max_dd,
            "current_drawdown_pct": current_dd,
            "win_rate_pct": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "win_loss_ratio": win_loss_ratio,
            "expectancy_per_trade": expectancy,
            "calmar_ratio": calmar,
            "trades_daily": frequency["daily"],
            "trades_weekly": frequency["weekly"],
            "trades_monthly": frequency["monthly"],
            "avg_hold_time_hours": avg_hold,
            "total_trades": len(self.state.trades),
            "current_equity": self.state.current_equity,
            "peak_equity": self.state.peak_equity,
            "total_pnl": sum(t.pnl for t in self.state.trades),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def get_symbol_performance(self) -> Dict[str, Dict[str, Any]]:
        symbol_stats: Dict[str, Dict[str, Any]] = {}
        for trade in self.state.trades:
            if trade.symbol not in symbol_stats:
                symbol_stats[trade.symbol] = {
                    "trades": 0,
                    "wins": 0,
                    "total_pnl": 0.0,
                    "avg_pnl_pct": 0.0,
                    "total_hold_time": 0
                }
            stats = symbol_stats[trade.symbol]
            stats["trades"] += 1
            if trade.pnl > 0:
                stats["wins"] += 1
            stats["total_pnl"] += trade.pnl
            stats["total_hold_time"] += trade.hold_time_seconds
        
        for symbol, stats in symbol_stats.items():
            if stats["trades"] > 0:
                stats["win_rate"] = round(stats["wins"] / stats["trades"] * 100, 2)
                stats["avg_hold_hours"] = round(stats["total_hold_time"] / stats["trades"] / 3600, 2)
        
        return symbol_stats

    def get_confidence_analysis(self) -> Dict[str, Any]:
        confidence_buckets = {
            "65-70": {"trades": 0, "wins": 0, "pnl": 0.0},
            "70-75": {"trades": 0, "wins": 0, "pnl": 0.0},
            "75-80": {"trades": 0, "wins": 0, "pnl": 0.0},
            "80-85": {"trades": 0, "wins": 0, "pnl": 0.0},
            "85-90": {"trades": 0, "wins": 0, "pnl": 0.0},
            "90-95": {"trades": 0, "wins": 0, "pnl": 0.0},
            "95-100": {"trades": 0, "wins": 0, "pnl": 0.0}
        }
        
        for trade in self.state.trades:
            conf = trade.confidence
            if conf < 70:
                bucket = "65-70"
            elif conf < 75:
                bucket = "70-75"
            elif conf < 80:
                bucket = "75-80"
            elif conf < 85:
                bucket = "80-85"
            elif conf < 90:
                bucket = "85-90"
            elif conf < 95:
                bucket = "90-95"
            else:
                bucket = "95-100"
            
            confidence_buckets[bucket]["trades"] += 1
            if trade.pnl > 0:
                confidence_buckets[bucket]["wins"] += 1
            confidence_buckets[bucket]["pnl"] += trade.pnl
        
        for bucket, stats in confidence_buckets.items():
            if stats["trades"] > 0:
                stats["win_rate"] = round(stats["wins"] / stats["trades"] * 100, 2)
                stats["avg_pnl"] = round(stats["pnl"] / stats["trades"], 2)
        
        return confidence_buckets
