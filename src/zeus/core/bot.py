#!/usr/bin/env python3
"""
ZEUS TRADING BOT - Main Trading Engine
Autonomous trading with real-time market scanning and order execution
"""

from __future__ import annotations
import asyncio
import os
import time
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path

from ..exchanges.kraken import KrakenExchange
from ..exchanges.base import OrderSide, OrderType, OHLCV
from ..indicators.prebreakout_detector import PreBreakoutDetector
from ..strategies.signal_generator import SignalGenerator, TradingSignal, StrategyMode
from ..strategies.risk_manager import RiskManager, RiskConfig
from ..alerts.telegram_bot import TelegramAlerts, AlertConfig
from .state import StateManager, TradeRecord

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Zeus.Bot")

KILL_SWITCH_FILE = "KILL_SWITCH"


def kill_switch_active() -> bool:
    return os.path.exists(KILL_SWITCH_FILE)


class ZeusBot:
    def __init__(
        self,
        kraken_key: str = "",
        kraken_secret: str = "",
        telegram_token: str = "",
        telegram_chat_id: str = "",
        mode: str = "PAPER"
    ):
        self.exchange = KrakenExchange(kraken_key, kraken_secret)
        self.state = StateManager()
        self.signal_gen = SignalGenerator()
        self.risk_manager = RiskManager()
        self.prebreakout = PreBreakoutDetector()
        self.telegram = TelegramAlerts(
            telegram_token, 
            telegram_chat_id,
            AlertConfig(enabled=bool(telegram_token and telegram_chat_id))
        )
        self.mode = mode
        self.running = False
        self._pairs_cache: List[str] = []
        self._last_pair_refresh = 0

    async def start(self) -> bool:
        logger.info(f"Starting Zeus Bot in {self.mode} mode...")
        if not await self.exchange.connect():
            logger.error("Failed to connect to Kraken")
            return False
        await self.telegram.initialize()
        await self.state.update_config(mode=self.mode)
        await self._refresh_balance()
        await self._refresh_pairs()
        await self.state.set_status("RUNNING")
        self.running = True
        await self.telegram.send_bot_status("RUNNING", self.mode, len(self._pairs_cache))
        logger.info(f"Zeus Bot started. Monitoring {len(self._pairs_cache)} pairs.")
        return True

    async def stop(self) -> None:
        logger.info("Stopping Zeus Bot...")
        self.running = False
        await self.state.set_status("STOPPED")
        await self.telegram.send_bot_status("STOPPED", self.mode, 0)
        await self.exchange.disconnect()
        logger.info("Zeus Bot stopped.")

    async def _refresh_balance(self) -> None:
        if self.mode == "PAPER":
            current_equity = self.state.state.equity
            if current_equity <= 0:
                await self.state.update_equity(10000.0)
                logger.info("Paper trading: Initialized with $10,000 virtual balance")
            else:
                logger.info(f"Paper trading balance: ${current_equity:.2f}")
            self.risk_manager.update_portfolio(self.state.state.equity, [])
            return
        try:
            balances = await self.exchange.fetch_balance()
            usd_balance = balances.get("USD", balances.get("ZUSD"))
            if usd_balance:
                equity = usd_balance.total
            else:
                equity = sum(b.total for b in balances.values())
            await self.state.update_equity(equity)
            holdings = {k: v.total for k, v in balances.items() if v.total > 0}
            await self.state.update_holdings(holdings)
            self.risk_manager.update_portfolio(equity, [])
            logger.info(f"Balance refreshed: ${equity:.2f}")
        except Exception as e:
            logger.error(f"Failed to refresh balance: {e}")

    async def _refresh_pairs(self) -> None:
        try:
            self._pairs_cache = await self.exchange.fetch_usd_pairs()
            self._last_pair_refresh = time.time()
            logger.info(f"Refreshed pairs list: {len(self._pairs_cache)} USD pairs")
        except Exception as e:
            logger.error(f"Failed to refresh pairs: {e}")

    async def scan_markets(self) -> List[Dict[str, Any]]:
        await self.state.set_status("SCANNING")
        logger.info(f"Scanning {len(self._pairs_cache)} pairs...")
        if time.time() - self._last_pair_refresh > 3600:
            await self._refresh_pairs()
        candidates = []
        semaphore = asyncio.Semaphore(10)
        async def analyze_pair(pair: str) -> Optional[Dict[str, Any]]:
            async with semaphore:
                try:
                    ohlcv = await self.exchange.fetch_ohlcv(pair, "1h", 100)
                    if len(ohlcv) < 50:
                        return None
                    high = [c.high for c in ohlcv]
                    low = [c.low for c in ohlcv]
                    close = [c.close for c in ohlcv]
                    volume = [c.volume for c in ohlcv]
                    analysis = await self.prebreakout.analyze(pair, high, low, close, volume)
                    if analysis.get("stage") in ["PRE_BREAKOUT", "BREAKOUT", "ACCUMULATION"]:
                        if analysis.get("prebreakout_score", 0) >= 50:
                            order_book = await self.exchange.analyze_order_book(pair)
                            return {
                                "symbol": pair,
                                "price": close[-1],
                                "prebreakout_score": analysis["prebreakout_score"],
                                "stage": analysis["stage"],
                                "buy_anchor": analysis["buy_anchor"],
                                "sell_anchor": analysis["sell_anchor"],
                                "stop_loss": analysis["stop_loss"],
                                "take_profit": analysis["take_profit"],
                                "atr": analysis["atr"],
                                "features": analysis["features"],
                                "reasons": analysis.get("reasons", []),
                                "spread_pct": order_book.get("spread_pct", 0) if order_book else 0,
                                "imbalance": order_book.get("imbalance", 0) if order_book else 0
                            }
                except Exception as e:
                    logger.debug(f"Error analyzing {pair}: {e}")
                return None
        results = await asyncio.gather(*[analyze_pair(p) for p in self._pairs_cache])
        candidates = [r for r in results if r is not None]
        candidates.sort(key=lambda x: x["prebreakout_score"], reverse=True)
        await self.state.set_candidates(candidates[:20])
        await self.state.set_status("IDLE")
        logger.info(f"Scan complete. Found {len(candidates)} candidates.")
        for c in candidates[:3]:
            if c["prebreakout_score"] >= 70:
                await self.telegram.send_prebreakout_alert(
                    c["symbol"],
                    c["prebreakout_score"],
                    c["stage"],
                    c["buy_anchor"],
                    c["sell_anchor"],
                    c["reasons"]
                )
        return candidates

    async def generate_signals(self, candidates: List[Dict[str, Any]]) -> List[TradingSignal]:
        signals = []
        strategy_map = {
            "scalp": StrategyMode.SCALP,
            "day_trade": StrategyMode.DAY_TRADE,
            "swing": StrategyMode.SWING,
            "pre_breakout": StrategyMode.PRE_BREAKOUT
        }
        mode = strategy_map.get(self.state.state.config.strategy_mode, StrategyMode.DAY_TRADE)
        for candidate in candidates[:10]:
            try:
                ohlcv = await self.exchange.fetch_ohlcv(candidate["symbol"], "15m", 100)
                if len(ohlcv) < 50:
                    continue
                order_book = await self.exchange.analyze_order_book(candidate["symbol"])
                signal = await self.signal_gen.generate_signal(
                    candidate["symbol"],
                    ohlcv,
                    mode,
                    order_book
                )
                if signal and signal.confidence >= self.state.state.config.min_confidence:
                    signals.append(signal)
                    await self.telegram.send_signal_alert(signal)
            except Exception as e:
                logger.error(f"Error generating signal for {candidate['symbol']}: {e}")
        signals.sort(key=lambda x: x.confidence, reverse=True)
        return signals

    async def execute_trade(self, signal: TradingSignal) -> Optional[str]:
        if not self.state.can_open_new_trade():
            logger.warning("Max positions reached, skipping trade")
            return None
        if signal.symbol in self.state.get_active_symbols():
            logger.warning(f"Already have position in {signal.symbol}")
            return None
        can_trade, reasons = self.risk_manager.can_open_position(signal.symbol)
        if not can_trade:
            logger.warning(f"Risk check failed: {reasons}")
            return None
        size = self.risk_manager.calculate_optimal_size(
            signal.entry_price,
            signal.stop_loss,
            signal.confidence
        )
        if size <= 0:
            logger.warning("Position size too small")
            return None
        max_size = self.state.state.config.per_trade_amount / signal.entry_price
        size = min(size, max_size)
        
        order_value = size * signal.entry_price
        min_order_value = 5.0
        if order_value < min_order_value:
            logger.warning(f"Order value ${order_value:.2f} below minimum ${min_order_value:.2f}, skipping {signal.symbol}")
            return None
        trade_id = f"T{int(time.time() * 1000)}"
        if self.mode == "LIVE":
            try:
                order = await self.exchange.create_order(
                    signal.symbol,
                    OrderType.MARKET,
                    signal.side,
                    size
                )
                if not order:
                    logger.error("Order placement failed")
                    return None
                trade_id = order.id
                logger.info(f"Order placed: {trade_id}")
            except Exception as e:
                logger.error(f"Order execution failed: {e}")
                await self.telegram.send_error_alert("Order Failed", str(e))
                return None
        trade = TradeRecord(
            id=trade_id,
            symbol=signal.symbol,
            side=signal.side.value,
            entry_price=signal.entry_price,
            size=size,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            status="open",
            entry_time=datetime.now(timezone.utc).isoformat(),
            strategy=signal.strategy_mode.value,
            confidence=signal.confidence,
            prebreakout_score=signal.prebreakout_score
        )
        await self.state.open_trade(trade)
        await self.telegram.send_trade_opened(
            signal.symbol,
            signal.side,
            signal.entry_price,
            size,
            signal.stop_loss,
            signal.take_profit
        )
        logger.info(f"Trade opened: {signal.symbol} {signal.side.value} @ {signal.entry_price}")
        return trade_id

    async def manage_positions(self) -> None:
        for trade_id, trade in list(self.state.state.active_trades.items()):
            try:
                ticker = await self.exchange.fetch_ticker(trade.symbol)
                if not ticker:
                    continue
                current_price = ticker.last
                should_close = False
                close_reason = ""
                if trade.side == "buy":
                    if current_price <= trade.stop_loss:
                        should_close = True
                        close_reason = "Stop Loss Hit"
                    elif current_price >= trade.take_profit:
                        should_close = True
                        close_reason = "Take Profit Hit"
                else:
                    if current_price >= trade.stop_loss:
                        should_close = True
                        close_reason = "Stop Loss Hit"
                    elif current_price <= trade.take_profit:
                        should_close = True
                        close_reason = "Take Profit Hit"
                if should_close:
                    await self._close_position(trade_id, current_price, close_reason)
            except Exception as e:
                logger.error(f"Error managing position {trade_id}: {e}")

    async def _close_position(self, trade_id: str, exit_price: float, reason: str) -> None:
        trade = self.state.state.active_trades.get(trade_id)
        if not trade:
            return
        if self.mode == "LIVE":
            try:
                side = OrderSide.SELL if trade.side == "buy" else OrderSide.BUY
                await self.exchange.create_order(
                    trade.symbol,
                    OrderType.MARKET,
                    side,
                    trade.size
                )
            except Exception as e:
                logger.error(f"Failed to close position: {e}")
                return
        closed_trade = await self.state.close_trade(trade_id, exit_price, reason)
        if closed_trade:
            await self.telegram.send_trade_closed(
                closed_trade.symbol,
                OrderSide(closed_trade.side),
                closed_trade.entry_price,
                exit_price,
                closed_trade.size,
                closed_trade.pnl,
                reason
            )
            self.risk_manager.record_trade(closed_trade.pnl, closed_trade.pnl >= 0)
            logger.info(f"Trade closed: {closed_trade.symbol} PnL: ${closed_trade.pnl:.2f}")

    async def run_cycle(self) -> None:
        try:
            await self._refresh_balance()
            candidates = await self.scan_markets()
            if candidates:
                signals = await self.generate_signals(candidates)
                for signal in signals[:3]:
                    if self.state.can_open_new_trade():
                        await self.execute_trade(signal)
            await self.manage_positions()
        except Exception as e:
            logger.error(f"Cycle error: {e}")
            await self.state.increment_errors()
            await self.telegram.send_error_alert("Cycle Error", str(e))

    async def run_forever(self) -> None:
        if not await self.start():
            return
        try:
            while self.running:
                if kill_switch_active():
                    logger.warning("Kill switch activated!")
                    break
                await self.run_cycle()
                interval = self.state.state.config.scan_interval
                logger.info(f"Sleeping for {interval} seconds...")
                await asyncio.sleep(interval)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
        finally:
            await self.stop()

    def get_status(self) -> Dict[str, Any]:
        return self.state.to_dict()


async def main():
    kraken_key = os.environ.get("KRAKEN_API_KEY", "")
    kraken_secret = os.environ.get("KRAKEN_API_SECRET", "")
    telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    mode = os.environ.get("TRADING_MODE", "PAPER")
    bot = ZeusBot(
        kraken_key=kraken_key,
        kraken_secret=kraken_secret,
        telegram_token=telegram_token,
        telegram_chat_id=telegram_chat_id,
        mode=mode
    )
    await bot.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
