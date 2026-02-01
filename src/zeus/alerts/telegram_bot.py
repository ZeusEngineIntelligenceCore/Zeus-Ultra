#!/usr/bin/env python3
"""
ZEUS TELEGRAM BOT - Real-time Trading Alerts with Learning
Sends notifications for trades, signals, and learns user preferences
"""

from __future__ import annotations
import asyncio
import logging
import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from pathlib import Path
import statistics

try:
    from telegram import Bot
    from telegram.error import TelegramError
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

from ..strategies.signal_generator import TradingSignal, SignalType
from ..exchanges.base import Position, OrderSide

logger = logging.getLogger("Zeus.Telegram")


@dataclass
class TelegramLearningState:
    preferred_symbols: Dict[str, float] = field(default_factory=dict)
    avoided_symbols: List[str] = field(default_factory=list)
    preferred_hours: List[int] = field(default_factory=list)
    min_profit_threshold: float = 3.0
    max_acceptable_loss: float = -2.0
    preferred_hold_duration_mins: int = 60
    alert_frequency_preference: str = "normal"
    successful_trade_patterns: List[Dict] = field(default_factory=list)
    interaction_history: List[Dict] = field(default_factory=list)
    total_alerts_sent: int = 0
    positive_feedback_count: int = 0
    negative_feedback_count: int = 0
    learned_confidence_threshold: float = 70.0
    best_performing_strategies: List[str] = field(default_factory=list)
    last_updated: str = ""


class TelegramLearningEngine:
    def __init__(self, data_file: str = "data/telegram_learning.json"):
        self.data_file = Path(data_file)
        self.state = TelegramLearningState()
        self._load_state()
        
    def _load_state(self) -> None:
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.state.preferred_symbols = data.get("preferred_symbols", {})
                    self.state.avoided_symbols = data.get("avoided_symbols", [])
                    self.state.preferred_hours = data.get("preferred_hours", [])
                    self.state.min_profit_threshold = data.get("min_profit_threshold", 3.0)
                    self.state.max_acceptable_loss = data.get("max_acceptable_loss", -2.0)
                    self.state.preferred_hold_duration_mins = data.get("preferred_hold_duration_mins", 60)
                    self.state.alert_frequency_preference = data.get("alert_frequency_preference", "normal")
                    self.state.successful_trade_patterns = data.get("successful_trade_patterns", [])[-100:]
                    self.state.interaction_history = data.get("interaction_history", [])[-200:]
                    self.state.total_alerts_sent = data.get("total_alerts_sent", 0)
                    self.state.positive_feedback_count = data.get("positive_feedback_count", 0)
                    self.state.negative_feedback_count = data.get("negative_feedback_count", 0)
                    self.state.learned_confidence_threshold = data.get("learned_confidence_threshold", 70.0)
                    self.state.best_performing_strategies = data.get("best_performing_strategies", [])
                    self.state.last_updated = data.get("last_updated", "")
                logger.info(f"Telegram Learning: Loaded state with {len(self.state.preferred_symbols)} preferred symbols")
            except Exception as e:
                logger.error(f"Failed to load Telegram learning state: {e}")
                
    def _save_state(self) -> None:
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.state.last_updated = datetime.now(timezone.utc).isoformat()
            data = asdict(self.state)
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save Telegram learning state: {e}")
            
    def record_trade_outcome(self, symbol: str, pnl: float, pnl_pct: float, 
                             hold_duration_mins: int, strategy: str, 
                             entry_hour: int) -> None:
        if pnl > 0:
            if symbol not in self.state.preferred_symbols:
                self.state.preferred_symbols[symbol] = 0.0
            self.state.preferred_symbols[symbol] += pnl_pct
            if symbol in self.state.avoided_symbols:
                self.state.avoided_symbols.remove(symbol)
            self.state.successful_trade_patterns.append({
                "symbol": symbol,
                "pnl_pct": pnl_pct,
                "hold_duration": hold_duration_mins,
                "strategy": strategy,
                "entry_hour": entry_hour,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            if entry_hour not in self.state.preferred_hours:
                win_hours = [p.get("entry_hour") for p in self.state.successful_trade_patterns 
                            if p.get("entry_hour") is not None]
                if win_hours:
                    hour_counts = {}
                    for h in win_hours:
                        hour_counts[h] = hour_counts.get(h, 0) + 1
                    self.state.preferred_hours = sorted(hour_counts.keys(), 
                                                       key=lambda x: hour_counts[x], 
                                                       reverse=True)[:8]
            if strategy and strategy not in self.state.best_performing_strategies:
                strat_wins = [p.get("strategy") for p in self.state.successful_trade_patterns 
                             if p.get("pnl_pct", 0) > 2.0]
                if strat_wins:
                    strat_counts = {}
                    for s in strat_wins:
                        if s:
                            strat_counts[s] = strat_counts.get(s, 0) + 1
                    self.state.best_performing_strategies = sorted(strat_counts.keys(),
                                                                   key=lambda x: strat_counts[x],
                                                                   reverse=True)[:5]
        else:
            if symbol in self.state.preferred_symbols:
                self.state.preferred_symbols[symbol] -= abs(pnl_pct)
                if self.state.preferred_symbols[symbol] < -5:
                    if symbol not in self.state.avoided_symbols:
                        self.state.avoided_symbols.append(symbol)
        win_pcts = [p.get("pnl_pct", 0) for p in self.state.successful_trade_patterns[-50:]]
        if len(win_pcts) >= 10:
            self.state.min_profit_threshold = max(2.0, statistics.median(win_pcts) * 0.8)
        hold_times = [p.get("hold_duration", 60) for p in self.state.successful_trade_patterns[-50:]]
        if len(hold_times) >= 10:
            self.state.preferred_hold_duration_mins = int(statistics.median(hold_times))
        self._save_state()
        
    def record_alert_sent(self, alert_type: str, symbol: str = "") -> None:
        self.state.total_alerts_sent += 1
        self.state.interaction_history.append({
            "type": "alert_sent",
            "alert_type": alert_type,
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        if len(self.state.interaction_history) > 200:
            self.state.interaction_history = self.state.interaction_history[-200:]
        self._save_state()
        
    def should_alert_for_symbol(self, symbol: str, confidence: float) -> bool:
        if symbol in self.state.avoided_symbols:
            return confidence > self.state.learned_confidence_threshold + 15
        if symbol in self.state.preferred_symbols:
            boost = min(10, self.state.preferred_symbols[symbol] / 2)
            return confidence > self.state.learned_confidence_threshold - boost
        return confidence >= self.state.learned_confidence_threshold
        
    def is_preferred_trading_hour(self) -> bool:
        if not self.state.preferred_hours:
            return True
        current_hour = datetime.now(timezone.utc).hour
        return current_hour in self.state.preferred_hours
        
    def get_optimal_profit_target(self) -> float:
        return self.state.min_profit_threshold
        
    def get_optimal_hold_duration(self) -> int:
        return self.state.preferred_hold_duration_mins
        
    def get_learning_insights(self) -> Dict[str, Any]:
        top_symbols = sorted(self.state.preferred_symbols.items(), 
                            key=lambda x: x[1], reverse=True)[:10]
        return {
            "top_symbols": dict(top_symbols),
            "avoided_symbols": self.state.avoided_symbols[:10],
            "preferred_hours": self.state.preferred_hours,
            "min_profit_threshold": self.state.min_profit_threshold,
            "optimal_hold_mins": self.state.preferred_hold_duration_mins,
            "confidence_threshold": self.state.learned_confidence_threshold,
            "best_strategies": self.state.best_performing_strategies,
            "total_alerts": self.state.total_alerts_sent,
            "successful_patterns": len(self.state.successful_trade_patterns)
        }
        
    def run_learning_cycle(self) -> Dict[str, Any]:
        patterns = self.state.successful_trade_patterns
        if len(patterns) < 5:
            return {"status": "insufficient_data"}
        win_pcts = [p.get("pnl_pct", 0) for p in patterns[-50:]]
        if win_pcts:
            avg_win = statistics.mean(win_pcts)
            self.state.min_profit_threshold = max(2.0, avg_win * 0.7)
        hold_times = [p.get("hold_duration", 60) for p in patterns[-50:]]
        if hold_times:
            self.state.preferred_hold_duration_mins = int(statistics.median(hold_times))
        if len(patterns) >= 20:
            high_profit_trades = [p for p in patterns if p.get("pnl_pct", 0) >= 3.0]
            if len(high_profit_trades) >= 5:
                hours = [p.get("entry_hour") for p in high_profit_trades if p.get("entry_hour") is not None]
                if hours:
                    hour_counts = {}
                    for h in hours:
                        hour_counts[h] = hour_counts.get(h, 0) + 1
                    self.state.preferred_hours = sorted(hour_counts.keys(),
                                                       key=lambda x: hour_counts[x],
                                                       reverse=True)[:8]
        self._save_state()
        logger.info(f"Telegram Learning: Cycle complete. Profit target: {self.state.min_profit_threshold:.1f}%")
        return {
            "status": "success",
            "min_profit_threshold": self.state.min_profit_threshold,
            "optimal_hold_mins": self.state.preferred_hold_duration_mins,
            "preferred_hours": self.state.preferred_hours
        }


@dataclass
class AlertConfig:
    enabled: bool = True
    send_signals: bool = False
    send_trades: bool = True
    send_portfolio_updates: bool = False
    send_daily_summary: bool = False
    send_bot_status: bool = False
    min_confidence_alert: float = 70.0
    alert_on_prebreakout: bool = False
    alert_on_stop_hit: bool = True
    alert_on_tp_hit: bool = True
    quiet_hours_start: int = 0
    quiet_hours_end: int = 0
    batch_interval_minutes: int = 30
    
    def enforce_trade_only_alerts(self) -> None:
        """Enforce that only trade open/close alerts are sent."""
        self.send_signals = False
        self.send_portfolio_updates = False
        self.send_daily_summary = False
        self.send_bot_status = False
        self.alert_on_prebreakout = False


@dataclass
class BatchedAlert:
    message: str
    timestamp: datetime
    alert_type: str


class TelegramAlerts:
    def __init__(self, token: str, chat_id: str, config: Optional[AlertConfig] = None):
        self.token = token
        self.chat_id = chat_id
        self.config = config or AlertConfig()
        self.bot: Optional[Bot] = None
        self._initialized = False
        self._message_queue: List[str] = []
        self._last_message_time: Optional[datetime] = None
        self._batched_alerts: List[BatchedAlert] = []
        self._last_batch_sent: Optional[datetime] = None
        self._urgent_types = {"trade_closed", "bot_status", "trade_opened"}
        self._error_rate_limit: Dict[str, datetime] = {}
        self._error_rate_limit_seconds = 300
        self._error_count_this_hour = 0
        self._error_count_reset_time: Optional[datetime] = None
        self._max_errors_per_hour = 5
        self.learning_engine = TelegramLearningEngine()

    async def initialize(self) -> bool:
        if not TELEGRAM_AVAILABLE:
            logger.warning("python-telegram-bot not installed")
            return False
        if not self.token or not self.chat_id:
            logger.warning("Telegram token or chat_id not configured")
            return False
        try:
            self.bot = Bot(token=self.token)
            me = await self.bot.get_me()
            logger.info(f"Telegram bot initialized: @{me.username}")
            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}")
            return False

    async def send_message(self, message: str, parse_mode: str = "HTML", urgent: bool = False, alert_type: str = "general") -> bool:
        if not self._initialized or not self.bot:
            self._message_queue.append(message)
            return False
        is_urgent = urgent or alert_type in self._urgent_types
        if not is_urgent:
            self._batched_alerts.append(BatchedAlert(
                message=message,
                timestamp=datetime.now(timezone.utc),
                alert_type=alert_type
            ))
            await self._check_and_send_batch()
            return True
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode
            )
            self._last_message_time = datetime.now(timezone.utc)
            return True
        except TelegramError as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    async def _check_and_send_batch(self) -> bool:
        if not self._batched_alerts:
            return False
        now = datetime.now(timezone.utc)
        if self._last_batch_sent:
            elapsed = (now - self._last_batch_sent).total_seconds() / 60
            if elapsed < self.config.batch_interval_minutes:
                return False
        return await self._send_batched_alerts()

    async def _send_batched_alerts(self) -> bool:
        if not self._batched_alerts or not self._initialized or not self.bot:
            return False
        alerts_by_type: Dict[str, List[BatchedAlert]] = {}
        for alert in self._batched_alerts:
            if alert.alert_type not in alerts_by_type:
                alerts_by_type[alert.alert_type] = []
            alerts_by_type[alert.alert_type].append(alert)
        summary_parts = [f"📋 <b>ZEUS ALERT SUMMARY</b> ({len(self._batched_alerts)} alerts)\n"]
        for alert_type, alerts in alerts_by_type.items():
            type_name = alert_type.replace("_", " ").title()
            summary_parts.append(f"\n<b>{type_name}</b> ({len(alerts)}):")
            for alert in alerts[-3:]:
                short_msg = alert.message.split('\n')[2] if len(alert.message.split('\n')) > 2 else alert.message[:100]
                summary_parts.append(f"• {short_msg.strip()}")
            if len(alerts) > 3:
                summary_parts.append(f"  ... and {len(alerts) - 3} more")
        summary_parts.append(f"\n⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
        summary = "\n".join(summary_parts)
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=summary,
                parse_mode="HTML"
            )
            self._last_batch_sent = datetime.now(timezone.utc)
            self._batched_alerts.clear()
            return True
        except TelegramError as e:
            logger.error(f"Failed to send batched alerts: {e}")
            return False

    async def force_send_batch(self) -> bool:
        return await self._send_batched_alerts()

    async def send_signal_alert(self, signal: TradingSignal) -> bool:
        if not self.config.send_signals:
            return False
        if signal.confidence < self.config.min_confidence_alert:
            return False
        emoji = self._get_signal_emoji(signal.signal_type)
        side_emoji = "🟢" if signal.side == OrderSide.BUY else "🔴"
        message = f"""
{emoji} <b>ZEUS SIGNAL ALERT</b> {emoji}

{side_emoji} <b>{signal.symbol}</b>
<b>Signal:</b> {signal.signal_type.value.upper()}
<b>Confidence:</b> {signal.confidence:.1f}%

💰 <b>Entry:</b> ${signal.entry_price:.8f}
🛑 <b>Stop Loss:</b> ${signal.stop_loss:.8f}
🎯 <b>Take Profit:</b> ${signal.take_profit:.8f}
📊 <b>R:R Ratio:</b> {signal.risk_reward:.2f}

<b>Strategy:</b> {signal.strategy_mode.value.upper()}
<b>Pre-Breakout Score:</b> {signal.prebreakout_score:.1f}%

<b>Reasons:</b>
{self._format_reasons(signal.reasons)}

⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
"""
        return await self.send_message(message.strip(), alert_type="signal")

    async def send_trade_opened(
        self,
        symbol: str,
        side: OrderSide,
        entry_price: float,
        size: float,
        stop_loss: float,
        take_profit: float
    ) -> bool:
        if not self.config.send_trades:
            return False
        side_emoji = "🟢 LONG" if side == OrderSide.BUY else "🔴 SHORT"
        message = f"""
📈 <b>TRADE OPENED</b>

{side_emoji} <b>{symbol}</b>

💵 <b>Entry:</b> ${entry_price:.8f}
📦 <b>Size:</b> {size:.8f}
💰 <b>Value:</b> ${entry_price * size:.2f}

🛑 <b>Stop Loss:</b> ${stop_loss:.8f}
🎯 <b>Take Profit:</b> ${take_profit:.8f}

⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
"""
        return await self.send_message(message.strip(), alert_type="trade_opened")

    async def send_trade_closed(
        self,
        symbol: str,
        side: OrderSide,
        entry_price: float,
        exit_price: float,
        size: float,
        pnl: float,
        reason: str
    ) -> bool:
        if not self.config.send_trades:
            return False
        pnl_emoji = "✅" if pnl >= 0 else "❌"
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        if side == OrderSide.SELL:
            pnl_pct = -pnl_pct
        message = f"""
{pnl_emoji} <b>TRADE CLOSED</b>

<b>{symbol}</b> {'LONG' if side == OrderSide.BUY else 'SHORT'}

💵 <b>Entry:</b> ${entry_price:.8f}
💵 <b>Exit:</b> ${exit_price:.8f}
📦 <b>Size:</b> {size:.8f}

{'💰' if pnl >= 0 else '💸'} <b>P&L:</b> ${pnl:+.2f} ({pnl_pct:+.2f}%)
📝 <b>Reason:</b> {reason}

⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
"""
        return await self.send_message(message.strip(), alert_type="trade_closed")

    async def send_prebreakout_alert(
        self,
        symbol: str,
        score: float,
        stage: str,
        buy_anchor: float,
        sell_anchor: float,
        reasons: List[str]
    ) -> bool:
        if not self.config.alert_on_prebreakout:
            return False
        if score < 70:
            return False
        message = f"""
🔥🔥🔥 <b>PRE-BREAKOUT DETECTED</b> 🔥🔥🔥

<b>{symbol}</b>

📊 <b>Score:</b> {score:.1f}%
🎯 <b>Stage:</b> {stage}

💚 <b>Buy Zone:</b> ${buy_anchor:.8f}
❤️ <b>Sell Zone:</b> ${sell_anchor:.8f}

<b>Signals:</b>
{self._format_reasons(reasons)}

⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
"""
        return await self.send_message(message.strip(), alert_type="prebreakout")

    async def send_portfolio_update(self, report: Dict[str, Any]) -> bool:
        if not self.config.send_portfolio_updates:
            return False
        streak_emoji = "🔥" if report.get("current_streak", 0) > 0 else "❄️"
        message = f"""
📊 <b>PORTFOLIO UPDATE</b>

💰 <b>Equity:</b> ${report.get('equity', 0):.2f}
📈 <b>Peak:</b> ${report.get('peak_equity', 0):.2f}
📉 <b>Drawdown:</b> {report.get('drawdown_pct', 0):.2f}%

📁 <b>Open Positions:</b> {report.get('open_positions', 0)}/{report.get('max_positions', 15)}
💵 <b>Unrealized P&L:</b> ${report.get('unrealized_pnl', 0):+.2f}
📅 <b>Daily P&L:</b> ${report.get('daily_pnl', 0):+.2f}

📊 <b>Statistics:</b>
• Total Trades: {report.get('total_trades', 0)}
• Win Rate: {report.get('win_rate', 0):.1f}%
• Profit Factor: {report.get('profit_factor', 0):.2f}
• Avg Win: ${report.get('avg_win', 0):.2f}
• Avg Loss: ${report.get('avg_loss', 0):.2f}

{streak_emoji} <b>Streak:</b> {report.get('current_streak', 0)}

⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
"""
        return await self.send_message(message.strip(), alert_type="portfolio_update")

    async def send_daily_summary(
        self,
        equity: float,
        daily_pnl: float,
        trades_today: int,
        wins: int,
        losses: int,
        top_winner: Optional[str] = None,
        top_loser: Optional[str] = None
    ) -> bool:
        if not self.config.send_daily_summary:
            return False
        pnl_emoji = "📈" if daily_pnl >= 0 else "📉"
        message = f"""
🌅 <b>DAILY SUMMARY</b>

💰 <b>Equity:</b> ${equity:.2f}
{pnl_emoji} <b>Daily P&L:</b> ${daily_pnl:+.2f}

📊 <b>Today's Trades:</b> {trades_today}
✅ <b>Wins:</b> {wins}
❌ <b>Losses:</b> {losses}
📈 <b>Win Rate:</b> {(wins/(wins+losses)*100) if (wins+losses) > 0 else 0:.1f}%

{f'🏆 <b>Top Winner:</b> {top_winner}' if top_winner else ''}
{f'💔 <b>Top Loser:</b> {top_loser}' if top_loser else ''}

⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d')}
"""
        return await self.send_message(message.strip(), alert_type="daily_summary")

    async def send_error_alert(self, error_type: str, message: str) -> bool:
        import re
        normalized_msg = re.sub(r'[0-9a-f]{8,}|[0-9]+\.[0-9]+|0x[0-9a-f]+', 'X', message[:100].lower())
        error_key = f"{error_type}:{normalized_msg}"
        now = datetime.now(timezone.utc)
        if self._error_count_reset_time is None or (now - self._error_count_reset_time).total_seconds() > 3600:
            self._error_count_reset_time = now
            self._error_count_this_hour = 0
        if self._error_count_this_hour >= self._max_errors_per_hour:
            logger.debug(f"Global error limit reached ({self._max_errors_per_hour}/hour)")
            return False
        if error_key in self._error_rate_limit:
            last_sent = self._error_rate_limit[error_key]
            if (now - last_sent).total_seconds() < self._error_rate_limit_seconds:
                logger.debug(f"Rate limiting error alert: {error_type}")
                return False
        self._error_rate_limit[error_key] = now
        self._error_count_this_hour += 1
        if len(self._error_rate_limit) > 100:
            oldest_keys = sorted(self._error_rate_limit.keys(), 
                                 key=lambda k: self._error_rate_limit[k])[:50]
            for k in oldest_keys:
                del self._error_rate_limit[k]
        alert = f"""
⚠️ <b>ZEUS ERROR ALERT</b>

<b>Type:</b> {error_type}
<b>Message:</b> {message}

⏰ {now.strftime('%Y-%m-%d %H:%M:%S')} UTC
"""
        return await self.send_message(alert.strip(), alert_type="error")

    async def send_bot_status(self, status: str, mode: str, pairs_count: int) -> bool:
        if not self.config.send_bot_status:
            return False
        status_emoji = "🟢" if status == "RUNNING" else "🔴" if status == "STOPPED" else "🟡"
        message = f"""
{status_emoji} <b>ZEUS BOT STATUS</b>

<b>Status:</b> {status}
<b>Mode:</b> {mode}
<b>Pairs Monitoring:</b> {pairs_count}

⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
"""
        return await self.send_message(message.strip(), alert_type="bot_status")

    def _get_signal_emoji(self, signal_type: SignalType) -> str:
        emoji_map = {
            SignalType.STRONG_BUY: "🚀🚀🚀",
            SignalType.BUY: "🚀🚀",
            SignalType.WEAK_BUY: "🚀",
            SignalType.NEUTRAL: "➡️",
            SignalType.WEAK_SELL: "📉",
            SignalType.SELL: "📉📉",
            SignalType.STRONG_SELL: "📉📉📉"
        }
        return emoji_map.get(signal_type, "📊")

    def _format_reasons(self, reasons: List[str]) -> str:
        if not reasons:
            return "• No specific signals"
        return "\n".join(f"• {r}" for r in reasons[:5])
