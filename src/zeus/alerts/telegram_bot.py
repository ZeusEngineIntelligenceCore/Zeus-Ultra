#!/usr/bin/env python3
"""
ZEUS TELEGRAM BOT - Real-time Trading Alerts
Sends notifications for trades, signals, and portfolio updates
"""

from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

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
        self._urgent_types = {"error", "trade_closed", "bot_status", "trade_opened"}

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
        alert = f"""
⚠️ <b>ZEUS ERROR ALERT</b>

<b>Type:</b> {error_type}
<b>Message:</b> {message}

⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
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
