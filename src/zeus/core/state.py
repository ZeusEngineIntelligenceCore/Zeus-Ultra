#!/usr/bin/env python3
"""
ZEUS STATE MANAGER - Centralized Trading State
Manages positions, trades, and bot configuration
"""

from __future__ import annotations
import asyncio
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

from ..exchanges.base import Position, OrderSide

logger = logging.getLogger("Zeus.State")


@dataclass
class TradeRecord:
    id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: Optional[float] = None
    size: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    pnl: float = 0.0
    status: str = "open"
    entry_time: str = ""
    exit_time: Optional[str] = None
    exit_reason: Optional[str] = None
    strategy: str = "day_trade"
    confidence: float = 0.0
    prebreakout_score: float = 0.0
    peak_price: float = 0.0
    peak_count: int = 0


@dataclass
class BotConfig:
    mode: str = "PAPER"
    max_open_positions: int = 2
    per_trade_amount: float = 15.0
    min_confidence: float = 80.0
    scan_interval: int = 30
    strategy_mode: str = "day_trade"
    enable_telegram: bool = True
    enable_trailing_stop: bool = True
    enable_partial_tp: bool = False
    risk_per_trade: float = 0.50
    max_daily_loss: float = 1.0


@dataclass
class BotState:
    status: str = "STOPPED"
    last_scan: Optional[str] = None
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    active_trades: Dict[str, TradeRecord] = field(default_factory=dict)
    trade_history: List[TradeRecord] = field(default_factory=list)
    equity: float = 0.0
    peak_equity: float = 0.0
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    wins: int = 0
    losses: int = 0
    errors: int = 0
    config: BotConfig = field(default_factory=BotConfig)
    holdings: Dict[str, float] = field(default_factory=dict)


class StateManager:
    def __init__(self, state_file: str = "data/bot_state.json"):
        self.state_file = Path(state_file)
        self.state = BotState()
        self.lock = asyncio.Lock()
        self._load_state()

    def _load_state(self) -> None:
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.state.config = BotConfig(**data.get("config", {}))
                    self.state.equity = data.get("equity", 0.0)
                    self.state.peak_equity = data.get("peak_equity", 0.0)
                    self.state.total_pnl = data.get("total_pnl", 0.0)
                    self.state.wins = data.get("wins", 0)
                    self.state.losses = data.get("losses", 0)
                    for trade_data in data.get("trade_history", []):
                        self.state.trade_history.append(TradeRecord(**trade_data))
                    for tid, trade_data in data.get("active_trades", {}).items():
                        self.state.active_trades[tid] = TradeRecord(**trade_data)
                logger.info(f"Loaded state from {self.state_file}")
            except Exception as e:
                logger.error(f"Failed to load state: {e}")

    async def save_state(self) -> None:
        async with self.lock:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                data = {
                    "config": asdict(self.state.config),
                    "equity": self.state.equity,
                    "peak_equity": self.state.peak_equity,
                    "total_pnl": self.state.total_pnl,
                    "wins": self.state.wins,
                    "losses": self.state.losses,
                    "active_trades": {k: asdict(v) for k, v in self.state.active_trades.items()},
                    "trade_history": [asdict(t) for t in self.state.trade_history[-100:]],
                    "last_updated": datetime.now(timezone.utc).isoformat()
                }
                with open(self.state_file, 'w') as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save state: {e}")

    async def update_config(self, **kwargs) -> None:
        async with self.lock:
            for key, value in kwargs.items():
                if hasattr(self.state.config, key):
                    setattr(self.state.config, key, value)
        await self.save_state()

    async def update_equity(self, equity: float) -> None:
        async with self.lock:
            self.state.equity = equity
            self.state.peak_equity = max(self.state.peak_equity, equity)
        await self.save_state()

    async def update_holdings(self, holdings: Dict[str, float]) -> None:
        async with self.lock:
            self.state.holdings = holdings

    async def add_candidate(self, candidate: Dict[str, Any]) -> None:
        async with self.lock:
            existing = [c for c in self.state.candidates if c.get("symbol") != candidate.get("symbol")]
            existing.append(candidate)
            existing.sort(key=lambda x: x.get("prebreakout_score", 0), reverse=True)
            self.state.candidates = existing[:20]

    async def set_candidates(self, candidates: List[Dict[str, Any]]) -> None:
        async with self.lock:
            self.state.candidates = candidates[:20]

    async def open_trade(self, trade: TradeRecord) -> None:
        async with self.lock:
            self.state.active_trades[trade.id] = trade
        await self.save_state()

    async def close_trade(self, trade_id: str, exit_price: float, reason: str) -> Optional[TradeRecord]:
        async with self.lock:
            if trade_id not in self.state.active_trades:
                return None
            trade = self.state.active_trades[trade_id]
            trade.exit_price = exit_price
            trade.exit_time = datetime.now(timezone.utc).isoformat()
            trade.exit_reason = reason
            trade.status = "closed"
            if trade.side == "buy":
                trade.pnl = (exit_price - trade.entry_price) * trade.size
            else:
                trade.pnl = (trade.entry_price - exit_price) * trade.size
            self.state.total_pnl += trade.pnl
            self.state.daily_pnl += trade.pnl
            if trade.pnl >= 0:
                self.state.wins += 1
            else:
                self.state.losses += 1
            self.state.trade_history.append(trade)
            del self.state.active_trades[trade_id]
        await self.save_state()
        return trade

    async def set_status(self, status: str) -> None:
        async with self.lock:
            self.state.status = status
            if status == "SCANNING":
                self.state.last_scan = datetime.now(timezone.utc).isoformat()

    async def increment_errors(self) -> None:
        async with self.lock:
            self.state.errors += 1

    async def reset_daily_stats(self) -> None:
        async with self.lock:
            self.state.daily_pnl = 0.0

    def get_open_positions_count(self) -> int:
        return len(self.state.active_trades)

    def can_open_new_trade(self) -> bool:
        return self.get_open_positions_count() < self.state.config.max_open_positions

    def get_active_symbols(self) -> List[str]:
        return [t.symbol for t in self.state.active_trades.values()]

    def get_stats(self) -> Dict[str, Any]:
        total_trades = self.state.wins + self.state.losses
        win_rate = (self.state.wins / total_trades * 100) if total_trades > 0 else 0
        winning_pnl = sum(t.pnl for t in self.state.trade_history if t.pnl > 0)
        losing_pnl = abs(sum(t.pnl for t in self.state.trade_history if t.pnl < 0))
        profit_factor = (winning_pnl / losing_pnl) if losing_pnl > 0 else 0
        drawdown = ((self.state.peak_equity - self.state.equity) / self.state.peak_equity * 100) if self.state.peak_equity > 0 else 0
        return {
            "equity": round(self.state.equity, 2),
            "peak_equity": round(self.state.peak_equity, 2),
            "daily_pnl": round(self.state.daily_pnl, 2),
            "total_pnl": round(self.state.total_pnl, 2),
            "wins": self.state.wins,
            "losses": self.state.losses,
            "total_trades": total_trades,
            "win_rate": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "drawdown": round(drawdown, 2),
            "open_positions": self.get_open_positions_count(),
            "max_positions": self.state.config.max_open_positions,
            "status": self.state.status,
            "mode": self.state.config.mode,
            "last_scan": self.state.last_scan
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.state.status,
            "config": asdict(self.state.config),
            "stats": self.get_stats(),
            "candidates": self.state.candidates,
            "active_trades": [asdict(t) for t in self.state.active_trades.values()],
            "trade_history": [asdict(t) for t in self.state.trade_history[-50:]],
            "holdings": self.state.holdings
        }
