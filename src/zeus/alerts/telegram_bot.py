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
    from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.error import TelegramError
    from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
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

    def get_daily_insights(self) -> Dict[str, Any]:
        patterns = self.state.successful_trade_patterns
        recent_patterns = patterns[-20:] if len(patterns) >= 20 else patterns
        total_pnl = sum(p.get("pnl_pct", 0) for p in recent_patterns)
        avg_pnl = total_pnl / len(recent_patterns) if recent_patterns else 0
        win_count = sum(1 for p in recent_patterns if p.get("pnl_pct", 0) > 0)
        win_rate = (win_count / len(recent_patterns) * 100) if recent_patterns else 0
        top_symbols = sorted(
            self.state.preferred_symbols.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        return {
            "recent_trades": len(recent_patterns),
            "avg_pnl_pct": avg_pnl,
            "win_rate": win_rate,
            "top_symbols": dict(top_symbols),
            "alerts_sent_today": self.state.total_alerts_sent,
            "confidence_threshold": self.state.learned_confidence_threshold,
            "best_strategies": self.state.best_performing_strategies[:3]
        }


@dataclass
class AlertConfig:
    enabled: bool = True
    send_signals: bool = False
    send_trades: bool = True
    send_portfolio_updates: bool = False
    send_daily_summary: bool = True
    send_bot_status: bool = False
    min_confidence_alert: float = 70.0
    alert_on_prebreakout: bool = False
    alert_on_stop_hit: bool = True
    alert_on_tp_hit: bool = True
    quiet_hours_start: int = 0
    quiet_hours_end: int = 0
    batch_interval_minutes: int = 30
    daily_report_hour: int = 20
    weekly_report_day: int = 6
    
    def enforce_trade_only_alerts(self) -> None:
        """Enforce that only trade open/close alerts are sent."""
        self.send_signals = False
        self.send_portfolio_updates = False
        self.send_daily_summary = True
        self.send_bot_status = False
        self.alert_on_prebreakout = False


@dataclass
class BatchedAlert:
    message: str
    timestamp: datetime
    alert_type: str


class TelegramAlerts:
    _polling_instance = None
    _polling_lock = None
    
    def __init__(self, token: str, chat_id: str, config: Optional[AlertConfig] = None):
        self.token = token
        self.chat_id = chat_id
        self.config = config or AlertConfig()
        self.bot: Optional[Bot] = None
        self.app: Optional[Any] = None
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
        self._bot_ref = None
        self._commands_registered = False
        self._polling_active = False

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
        projected_profit_pct = ((take_profit - entry_price) / entry_price) * 100 if take_profit > entry_price else 0
        profit_rating = "🚀🚀🚀" if projected_profit_pct >= 100 else "🚀🚀" if projected_profit_pct >= 50 else "🚀" if projected_profit_pct >= 30 else "📈" if projected_profit_pct >= 15 else ""
        message = f"""
📈 <b>TRADE OPENED</b>

{side_emoji} <b>{symbol}</b>

💵 <b>Entry:</b> ${entry_price:.8f}
📦 <b>Size:</b> {size:.8f}
💰 <b>Value:</b> ${entry_price * size:.2f}
{profit_rating} <b>Projected:</b> +{projected_profit_pct:.1f}%

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

    def set_bot_reference(self, bot_instance) -> None:
        self._bot_ref = bot_instance
        logger.info("Telegram: Bot reference set for interactive commands")

    def get_dashboard_url(self) -> str:
        import os
        domain = os.environ.get('REPLIT_DEV_DOMAIN', os.environ.get('REPLIT_DOMAINS', ''))
        if domain:
            return f"https://{domain}/dashboard"
        return "https://replit.com"

    def get_main_keyboard(self) -> 'InlineKeyboardMarkup':
        keyboard = [
            [
                InlineKeyboardButton("🌐 Dashboard", url=self.get_dashboard_url()),
                InlineKeyboardButton("📊 Status", callback_data="cmd_status")
            ],
            [
                InlineKeyboardButton("💼 Portfolio", callback_data="cmd_portfolio"),
                InlineKeyboardButton("📈 Trades", callback_data="cmd_trades")
            ],
            [
                InlineKeyboardButton("🎯 Candidates", callback_data="cmd_candidates"),
                InlineKeyboardButton("📉 Performance", callback_data="cmd_performance")
            ],
            [
                InlineKeyboardButton("⚙️ Settings", callback_data="cmd_settings"),
                InlineKeyboardButton("❓ Help", callback_data="cmd_help")
            ],
            [
                InlineKeyboardButton("🔄 Refresh", callback_data="cmd_refresh")
            ]
        ]
        return InlineKeyboardMarkup(keyboard)

    async def handle_command_status(self) -> str:
        if not self._bot_ref:
            return "⚠️ Bot reference not available"
        try:
            bot = self._bot_ref
            state_mgr = bot.state
            bot_state = state_mgr.state if hasattr(state_mgr, 'state') else state_mgr
            mode = "🔴 LIVE" if bot.mode == "LIVE" else "🟡 PAPER"
            status_str = bot_state.status if hasattr(bot_state, 'status') else "UNKNOWN"
            if status_str in ["SCANNING", "RUNNING"] or bot.running:
                running = "🟢 Running"
            else:
                running = "🔴 Stopped"
            balance = bot.exchange.cached_balance if hasattr(bot.exchange, 'cached_balance') else None
            balance_str = f"${balance:.2f}" if isinstance(balance, (int, float)) else "N/A"
            positions = state_mgr.get_open_positions_count() if hasattr(state_mgr, 'get_open_positions_count') else len(bot_state.active_trades) if hasattr(bot_state, 'active_trades') else 0
            holdings = len(bot_state.holdings) if hasattr(bot_state, 'holdings') else 0
            pairs_count = len(bot._pairs_cache) if hasattr(bot, '_pairs_cache') and bot._pairs_cache else "Initializing..."
            trading_cycle = bot_state.trading_cycle if hasattr(bot_state, 'trading_cycle') else 0
            last_scan = bot_state.last_scan if hasattr(bot_state, 'last_scan') and bot_state.last_scan else None
            last_scan_str = last_scan[:19] if last_scan else "N/A"
            msg = f"""
⚡ <b>ZEUS STATUS</b>

<b>Mode:</b> {mode}
<b>Status:</b> {running} ({status_str})
<b>Balance:</b> {balance_str}
<b>Open Positions:</b> {positions}
<b>Holdings:</b> {holdings} tokens
<b>Pairs Monitored:</b> {pairs_count}
<b>Trading Cycle:</b> {trading_cycle}
<b>Last Scan:</b> {last_scan_str}

⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
"""
            return msg.strip()
        except Exception as e:
            logger.error(f"Status command error: {e}")
            return f"⚠️ Error getting status: {str(e)[:100]}"

    async def handle_command_portfolio(self) -> str:
        if not self._bot_ref:
            return "⚠️ Bot reference not available"
        try:
            bot = self._bot_ref
            state_mgr = bot.state
            bot_state = state_mgr.state if hasattr(state_mgr, 'state') else state_mgr
            holdings = bot_state.holdings if hasattr(bot_state, 'holdings') else {}
            if not holdings:
                return "💼 <b>Portfolio</b>\n\nNo holdings found."
            lines = ["💼 <b>PORTFOLIO HOLDINGS</b>\n"]
            sorted_holdings = sorted(holdings.items(), key=lambda x: x[1], reverse=True)[:15]
            for symbol, amount in sorted_holdings:
                if amount > 0.0001:
                    lines.append(f"• <b>{symbol}:</b> {amount:.4f}")
            lines.append(f"\n<b>Total Holdings:</b> {len(holdings)} tokens")
            lines.append(f"\n⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Portfolio command error: {e}")
            return f"⚠️ Error getting portfolio: {str(e)[:100]}"

    async def handle_command_trades(self) -> str:
        if not self._bot_ref:
            return "⚠️ Bot reference not available"
        try:
            bot = self._bot_ref
            state_mgr = bot.state
            active_trades = state_mgr.state.active_trades if hasattr(state_mgr, 'state') else {}
            if not active_trades:
                return "📈 <b>Active Trades</b>\n\nNo open positions."
            lines = ["📈 <b>ACTIVE TRADES</b>\n"]
            for trade_id, trade in list(active_trades.items())[:10]:
                symbol = trade.symbol if hasattr(trade, 'symbol') else trade_id
                entry = trade.entry_price if hasattr(trade, 'entry_price') else 0
                current = trade.peak_price if hasattr(trade, 'peak_price') and trade.peak_price > 0 else entry
                pnl_pct = ((current - entry) / entry * 100) if entry > 0 else 0
                side = trade.side if hasattr(trade, 'side') else 'buy'
                side_emoji = "🟢" if side == 'buy' else "🔴"
                pnl_emoji = "📈" if pnl_pct >= 0 else "📉"
                lines.append(f"{side_emoji} <b>{symbol}:</b> {pnl_emoji} {pnl_pct:+.2f}% @ ${entry:.6f}")
            if len(active_trades) > 10:
                lines.append(f"\n... and {len(active_trades) - 10} more")
            lines.append(f"\n<b>Total Open:</b> {len(active_trades)} positions")
            lines.append(f"\n⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Trades command error: {e}")
            return f"⚠️ Error getting trades: {str(e)[:100]}"

    async def handle_command_candidates(self) -> str:
        if not self._bot_ref:
            return "⚠️ Bot reference not available"
        try:
            bot = self._bot_ref
            state = bot.state
            candidates = state.top_candidates if hasattr(state, 'top_candidates') else []
            if not candidates:
                return "🎯 <b>Top Candidates</b>\n\nNo candidates identified yet."
            lines = ["🎯 <b>TOP TRADING CANDIDATES</b>\n"]
            for i, c in enumerate(candidates[:10], 1):
                symbol = c.get('symbol', 'N/A')
                score = c.get('score', 0)
                confidence = c.get('confidence', 0)
                lines.append(f"{i}. <b>{symbol}</b> - Score: {score:.1f} | Conf: {confidence:.0f}%")
            lines.append(f"\n<b>Total Candidates:</b> {len(candidates)}")
            lines.append(f"\n⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Candidates command error: {e}")
            return f"⚠️ Error getting candidates: {str(e)[:100]}"

    async def handle_command_performance(self) -> str:
        if not self._bot_ref:
            return "⚠️ Bot reference not available"
        try:
            bot = self._bot_ref
            learning = self.learning_engine
            ml = bot.ml_engine if hasattr(bot, 'ml_engine') else None
            state = learning.state
            lines = ["📉 <b>PERFORMANCE ANALYTICS</b>\n"]
            lines.append(f"<b>Preferred Symbols:</b> {len(state.preferred_symbols)}")
            lines.append(f"<b>Avoided Symbols:</b> {len(state.avoided_symbols)}")
            lines.append(f"<b>Profit Threshold:</b> {state.min_profit_threshold}%")
            lines.append(f"<b>Confidence Level:</b> {state.learned_confidence_threshold:.0f}%")
            lines.append(f"<b>Alerts Sent:</b> {state.total_alerts_sent}")
            if state.best_performing_strategies:
                lines.append(f"\n<b>Best Strategies:</b>")
                for strat in state.best_performing_strategies[:3]:
                    lines.append(f"• {strat}")
            if ml and hasattr(ml, 'state'):
                ml_state = ml.state
                profiles = len(ml_state.symbol_profiles)
                cycles = ml_state.total_learning_cycles
                lines.append(f"\n<b>ML Engine:</b>")
                lines.append(f"• Symbol Profiles: {profiles}")
                lines.append(f"• Learning Cycles: {cycles}")
            lines.append(f"\n⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Performance command error: {e}")
            return f"⚠️ Error getting performance: {str(e)[:100]}"

    async def handle_command_settings(self) -> str:
        try:
            state = self.learning_engine.state
            lines = ["⚙️ <b>ZEUS SETTINGS</b>\n"]
            lines.append(f"<b>Alert Frequency:</b> {state.alert_frequency_preference}")
            lines.append(f"<b>Batch Interval:</b> {self.config.batch_interval_minutes} min")
            lines.append(f"<b>Min Profit Target:</b> {state.min_profit_threshold}%")
            lines.append(f"<b>Max Loss Threshold:</b> {state.max_acceptable_loss}%")
            lines.append(f"<b>Hold Duration Target:</b> {state.preferred_hold_duration_mins} min")
            lines.append(f"<b>Confidence Threshold:</b> {state.learned_confidence_threshold:.0f}%")
            lines.append(f"<b>Error Limit:</b> {self._max_errors_per_hour}/hour")
            lines.append(f"\n<i>Settings adjust automatically based on your trading patterns.</i>")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Settings command error: {e}")
            return f"⚠️ Error getting settings: {str(e)[:100]}"

    async def handle_command_help(self) -> str:
        return """
❓ <b>ZEUS COMMAND HELP</b>

<b>Available Commands:</b>

/dashboard - Open web dashboard
/analyze &lt;symbol&gt; - Full 30 KPI analysis
/status - View bot status & stats
/portfolio - View token holdings
/trades - View active trades
/candidates - View top trading candidates
/performance - View performance analytics
/settings - View current settings
/report - Get performance report
/help - Show this help message

<b>Analyzer Examples:</b>
/analyze BTC - Analyze Bitcoin
/analyze FIS - Analyze FIS token

<b>Quick Actions:</b>
Use the buttons below messages for quick navigation.

<b>Automatic Features:</b>
• Daily performance summaries
• Trade open/close alerts
• Pre-breakout signal detection
• ML-powered recommendations

⚡ Zeus - Your Autonomous Trading Partner
"""

    async def handle_callback_query(self, update: 'Update', context: 'ContextTypes.DEFAULT_TYPE') -> None:
        query = update.callback_query
        await query.answer()
        callback_data = query.data
        response = ""
        if callback_data == "cmd_status":
            response = await self.handle_command_status()
        elif callback_data == "cmd_portfolio":
            response = await self.handle_command_portfolio()
        elif callback_data == "cmd_trades":
            response = await self.handle_command_trades()
        elif callback_data == "cmd_candidates":
            response = await self.handle_command_candidates()
        elif callback_data == "cmd_performance":
            response = await self.handle_command_performance()
        elif callback_data == "cmd_settings":
            response = await self.handle_command_settings()
        elif callback_data == "cmd_help":
            response = await self.handle_command_help()
        elif callback_data == "cmd_refresh":
            response = "🔄 Refreshing..."
            await query.edit_message_text(text=response, parse_mode="HTML")
            response = await self.handle_command_status()
        if response:
            try:
                await query.edit_message_text(
                    text=response,
                    parse_mode="HTML",
                    reply_markup=self.get_main_keyboard()
                )
            except Exception as e:
                logger.error(f"Callback error: {e}")

    async def cmd_start(self, update: 'Update', context: 'ContextTypes.DEFAULT_TYPE') -> None:
        welcome = """
⚡ <b>Welcome to ZEUS Trading Bot!</b>

I'm your autonomous cryptocurrency trading assistant.

Use the buttons below or type commands to interact with me.

<b>Quick Start:</b>
• /status - Check bot status
• /help - View all commands
"""
        await update.message.reply_text(
            welcome.strip(),
            parse_mode="HTML",
            reply_markup=self.get_main_keyboard()
        )

    async def cmd_status(self, update: 'Update', context: 'ContextTypes.DEFAULT_TYPE') -> None:
        response = await self.handle_command_status()
        await update.message.reply_text(response, parse_mode="HTML", reply_markup=self.get_main_keyboard())

    async def cmd_portfolio(self, update: 'Update', context: 'ContextTypes.DEFAULT_TYPE') -> None:
        response = await self.handle_command_portfolio()
        await update.message.reply_text(response, parse_mode="HTML", reply_markup=self.get_main_keyboard())

    async def cmd_trades(self, update: 'Update', context: 'ContextTypes.DEFAULT_TYPE') -> None:
        response = await self.handle_command_trades()
        await update.message.reply_text(response, parse_mode="HTML", reply_markup=self.get_main_keyboard())

    async def cmd_candidates(self, update: 'Update', context: 'ContextTypes.DEFAULT_TYPE') -> None:
        response = await self.handle_command_candidates()
        await update.message.reply_text(response, parse_mode="HTML", reply_markup=self.get_main_keyboard())

    async def cmd_performance(self, update: 'Update', context: 'ContextTypes.DEFAULT_TYPE') -> None:
        response = await self.handle_command_performance()
        await update.message.reply_text(response, parse_mode="HTML", reply_markup=self.get_main_keyboard())

    async def cmd_settings(self, update: 'Update', context: 'ContextTypes.DEFAULT_TYPE') -> None:
        response = await self.handle_command_settings()
        await update.message.reply_text(response, parse_mode="HTML", reply_markup=self.get_main_keyboard())

    async def cmd_help(self, update: 'Update', context: 'ContextTypes.DEFAULT_TYPE') -> None:
        response = await self.handle_command_help()
        await update.message.reply_text(response, parse_mode="HTML", reply_markup=self.get_main_keyboard())

    async def cmd_report(self, update: 'Update', context: 'ContextTypes.DEFAULT_TYPE') -> None:
        if not self._bot_ref:
            await update.message.reply_text("⚠️ Bot reference not available")
            return
        try:
            state_mgr = self._bot_ref.state
            bot_state = state_mgr.state if hasattr(state_mgr, 'state') else state_mgr
            equity = self._bot_ref.exchange.cached_balance if hasattr(self._bot_ref.exchange, 'cached_balance') else 0.0
            if not isinstance(equity, (int, float)):
                equity = 0.0
            daily_pnl = bot_state.daily_pnl if hasattr(bot_state, 'daily_pnl') else 0.0
            active_trades = bot_state.active_trades if hasattr(bot_state, 'active_trades') else {}
            trades_today = len(active_trades)
            wins = bot_state.wins if hasattr(bot_state, 'wins') else 0
            losses = bot_state.losses if hasattr(bot_state, 'losses') else 0
            await self.send_daily_summary(
                equity=equity,
                daily_pnl=daily_pnl,
                trades_today=trades_today,
                wins=wins,
                losses=losses
            )
            await update.message.reply_text("📊 Report sent!", reply_markup=self.get_main_keyboard())
        except Exception as e:
            logger.error(f"Report command error: {e}")
            await update.message.reply_text(f"⚠️ Error: {str(e)[:100]}")

    async def cmd_dashboard(self, update: 'Update', context: 'ContextTypes.DEFAULT_TYPE') -> None:
        dashboard_url = self.get_dashboard_url()
        keyboard = [[InlineKeyboardButton("🌐 Open Dashboard", url=dashboard_url)]]
        msg = f"""
🌐 <b>ZEUS WEB DASHBOARD</b>

Access the full interactive dashboard to:
• View real-time portfolio stats
• Monitor active trades
• See top candidates
• Control bot settings

<b>URL:</b> {dashboard_url}

Tap the button below to open:
"""
        await update.message.reply_text(
            msg.strip(),
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    async def cmd_analyze(self, update: 'Update', context: 'ContextTypes.DEFAULT_TYPE') -> None:
        if not context.args:
            await update.message.reply_text(
                "🔬 <b>Coin Analyzer</b>\n\nUsage: /analyze &lt;symbol&gt;\n\nExample: /analyze BTC\n\nThis will run a full 30 KPI analysis across 5 timeframes.",
                parse_mode="HTML",
                reply_markup=self.get_main_keyboard()
            )
            return
        
        symbol = context.args[0].upper().strip()
        if not symbol.endswith("USD"):
            symbol = f"{symbol}USD"
        
        await update.message.reply_text(f"🔬 Analyzing <b>{symbol}</b>...\n\nRunning 30 KPIs across 5 timeframes. Please wait...", parse_mode="HTML")
        
        if not self._bot_ref:
            await update.message.reply_text("⚠️ Bot not available for analysis", reply_markup=self.get_main_keyboard())
            return
        
        try:
            bot = self._bot_ref
            exchange = bot.exchange
            prebreakout = bot.prebreakout
            
            ohlcv = await exchange.fetch_ohlcv(symbol, "15m", limit=500)
            if not ohlcv or len(ohlcv) < 50:
                await update.message.reply_text(
                    f"⚠️ Insufficient data for {symbol}. Make sure it's a valid Kraken trading pair.",
                    reply_markup=self.get_main_keyboard()
                )
                return
            
            high = [c.high for c in ohlcv]
            low = [c.low for c in ohlcv]
            close = [c.close for c in ohlcv]
            volume = [c.volume for c in ohlcv]
            
            analysis = await prebreakout.analyze(symbol, high, low, close, volume)
            
            score = analysis.get("prebreakout_score", 0)
            stage = analysis.get("stage", "UNKNOWN")
            confidence = analysis.get("confidence", 0)
            price = analysis.get("current_price", 0)
            features = analysis.get("features", {})
            reasons = analysis.get("reasons", [])
            
            stage_emoji = {
                "BREAKOUT": "🚀",
                "LATE_PRE-BREAKOUT": "🔥",
                "PRE-BREAKOUT": "⚡",
                "EARLY_SETUP": "📊",
                "ACCUMULATION": "🔄",
                "DORMANT": "💤"
            }.get(stage, "📈")
            
            msg_lines = [
                f"🔬 <b>ZEUS COIN ANALYSIS</b>",
                f"━━━━━━━━━━━━━━━━━━━━",
                f"",
                f"<b>Symbol:</b> {symbol}",
                f"<b>Price:</b> ${price:.8f}" if price < 1 else f"<b>Price:</b> ${price:.4f}",
                f"<b>Stage:</b> {stage_emoji} {stage}",
                f"<b>Score:</b> {score:.1f}/100",
                f"<b>Confidence:</b> {confidence:.1f}%",
                f"",
                f"━━━ <b>TOP KPIs</b> ━━━"
            ]
            
            top_kpis = [
                ("RSI", features.get("rsi", 0)),
                ("Momentum", features.get("momentum_cf", 0)),
                ("Pressure", features.get("pressure", 0)),
                ("Impulse", features.get("impulse", 0)),
                ("Squeeze", features.get("squeeze", 0)),
                ("ADX", features.get("adx_strength", 0)),
                ("Supertrend", features.get("supertrend_conf", 0)),
                ("Aroon", features.get("aroon_signal", 0)),
                ("Williams %R", features.get("williams_r", 0)),
                ("MFI", features.get("mfi_signal", 0)),
                ("OBV", features.get("obv_trend", 0)),
                ("Fibonacci", features.get("fibonacci_level", 0))
            ]
            
            for name, val in top_kpis:
                pct = val * 100
                emoji = "🟢" if pct >= 60 else "🟡" if pct >= 40 else "🔴"
                bar = "█" * int(val * 10) + "░" * (10 - int(val * 10))
                msg_lines.append(f"{emoji} {name}: {bar} {pct:.0f}%")
            
            if reasons:
                msg_lines.append("")
                msg_lines.append("━━━ <b>SIGNALS</b> ━━━")
                for reason in reasons[:5]:
                    msg_lines.append(f"✅ {reason}")
            
            msg_lines.extend([
                "",
                f"━━━ <b>TRADE LEVELS</b> ━━━",
                f"🎯 Entry: ${analysis.get('buy_anchor', 0):.8f}" if price < 1 else f"🎯 Entry: ${analysis.get('buy_anchor', 0):.4f}",
                f"🛑 Stop: ${analysis.get('stop_loss', 0):.8f}" if price < 1 else f"🛑 Stop: ${analysis.get('stop_loss', 0):.4f}",
                f"💰 Target: ${analysis.get('take_profit', 0):.8f}" if price < 1 else f"💰 Target: ${analysis.get('take_profit', 0):.4f}",
                "",
                f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"
            ])
            
            message = "\n".join(msg_lines)
            
            miniapp_url = self.get_dashboard_url().replace('/dashboard', '/miniapp')
            keyboard = [[InlineKeyboardButton("🔬 Open Analyzer", url=miniapp_url)]]
            
            await update.message.reply_text(
                message,
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            
        except Exception as e:
            logger.error(f"Analyze command error: {e}")
            await update.message.reply_text(
                f"⚠️ Error analyzing {symbol}: {str(e)[:100]}",
                reply_markup=self.get_main_keyboard()
            )

    async def start_command_listener(self) -> None:
        if not TELEGRAM_AVAILABLE or self._commands_registered:
            return
        logger.info("Telegram operating in send-only mode (alerts enabled, commands via web dashboard)")
        self._commands_registered = True
        self._polling_active = False
        TelegramAlerts._polling_instance = self

    async def _register_bot_commands(self) -> None:
        try:
            if TELEGRAM_AVAILABLE and self.app and self.app.bot:
                from telegram import BotCommand
                commands = [
                    BotCommand("start", "Start Zeus Bot and show main menu"),
                    BotCommand("dashboard", "Open web dashboard"),
                    BotCommand("analyze", "Analyze any coin - /analyze BTC"),
                    BotCommand("status", "View current bot status and balance"),
                    BotCommand("portfolio", "View your token holdings"),
                    BotCommand("trades", "View active open trades"),
                    BotCommand("candidates", "View top trading candidates"),
                    BotCommand("performance", "View trading performance stats"),
                    BotCommand("settings", "View and manage bot settings"),
                    BotCommand("report", "Generate performance report"),
                    BotCommand("help", "Show available commands")
                ]
                await self.app.bot.set_my_commands(commands)
                logger.info(f"Registered {len(commands)} bot commands with Telegram")
                await self.app.bot.set_my_description("Zeus Autonomous Trading Bot - Professional crypto trading with ML-powered signals, real-time alerts, and advanced risk management.")
                await self.app.bot.set_my_short_description("Zeus Trading Bot - Autonomous crypto trading")
                logger.info("Updated bot description")
        except Exception as e:
            logger.warning(f"Could not register bot commands: {e}")

    async def stop_command_listener(self) -> None:
        self._polling_active = False
        TelegramAlerts._polling_instance = None
        if self.app:
            try:
                if self.app.updater and self.app.updater.running:
                    await self.app.updater.stop()
                if self.app.running:
                    await self.app.stop()
                await self.app.shutdown()
                self.app = None
                self._commands_registered = False
                logger.info("Telegram command listener stopped and cleaned up")
            except Exception as e:
                logger.error(f"Error stopping command listener: {e}")
                self.app = None
                self._commands_registered = False

    async def check_scheduled_reports(self) -> bool:
        if not self.config.send_daily_summary or not self._bot_ref:
            return False
        try:
            from pytz import timezone as tz
            la_tz = tz('America/Los_Angeles')
            now = datetime.now(la_tz)
            current_hour = now.hour
            current_minute = now.minute
            current_weekday = now.weekday()
            last_report_key = "last_daily_report"
            last_report = getattr(self, '_last_daily_report', None)
            if last_report and last_report.date() == now.date():
                return False
            if current_hour == self.config.daily_report_hour and current_minute < 15:
                await self.send_scheduled_report(is_weekly=(current_weekday == self.config.weekly_report_day))
                self._last_daily_report = now
                return True
            return False
        except Exception as e:
            logger.error(f"Scheduled report check error: {e}")
            return False

    async def send_scheduled_report(self, is_weekly: bool = False) -> bool:
        if not self._bot_ref:
            return False
        try:
            bot = self._bot_ref
            state = bot.state
            from pytz import timezone as tz
            la_tz = tz('America/Los_Angeles')
            now = datetime.now(la_tz)
            report_type = "WEEKLY" if is_weekly else "DAILY"
            lines = [f"📊 <b>ZEUS {report_type} REPORT</b>", f"📅 {now.strftime('%A, %B %d, %Y')}\n"]
            open_positions = len(state.state.active_trades) if hasattr(state, 'state') else 0
            holdings_count = len(state.state.holdings) if hasattr(state, 'state') and hasattr(state.state, 'holdings') else 0
            balance = bot.exchange.cached_balance if hasattr(bot.exchange, 'cached_balance') else None
            balance_str = f"${balance:.2f}" if isinstance(balance, (int, float)) else "N/A"
            lines.append(f"<b>Account Status:</b>")
            lines.append(f"• Balance: {balance_str}")
            lines.append(f"• Open Positions: {open_positions}")
            lines.append(f"• Holdings: {holdings_count} tokens\n")
            ml = bot.ml_engine if hasattr(bot, 'ml_engine') else None
            if ml and hasattr(ml, 'state'):
                ml_state = ml.state
                if ml_state.trade_history_for_learning:
                    recent = ml_state.trade_history_for_learning[-20:]
                    wins = sum(1 for t in recent if t.get("pnl", 0) > 0)
                    total_pnl = sum(t.get("pnl", 0) for t in recent)
                    lines.append(f"<b>Recent Performance:</b>")
                    lines.append(f"• Last 20 Trades Win Rate: {wins/len(recent)*100:.1f}%")
                    lines.append(f"• Recent PnL: ${total_pnl:.2f}")
                    lines.append(f"• Total Trades Tracked: {len(ml_state.trade_history_for_learning)}")
                    lines.append(f"• Learning Cycles: {ml_state.total_learning_cycles}\n")
            learning = self.learning_engine.state
            if learning.preferred_symbols:
                top_symbols = sorted(learning.preferred_symbols.items(), key=lambda x: x[1], reverse=True)[:5]
                lines.append(f"<b>Top Performing Symbols:</b>")
                for sym, score in top_symbols:
                    lines.append(f"• {sym}: {score:.1f}")
                lines.append("")
            if learning.best_performing_strategies:
                lines.append(f"<b>Best Strategies:</b>")
                for strat in learning.best_performing_strategies[:3]:
                    lines.append(f"• {strat}")
                lines.append("")
            lines.append(f"⏰ Report generated at {now.strftime('%I:%M %p')} PT")
            lines.append("\n<i>Use /status for real-time updates</i>")
            message = "\n".join(lines)
            await self.send_message(message, alert_type="scheduled_report", urgent=True)
            logger.info(f"Sent {report_type.lower()} report")
            return True
        except Exception as e:
            logger.error(f"Failed to send scheduled report: {e}")
            return False
