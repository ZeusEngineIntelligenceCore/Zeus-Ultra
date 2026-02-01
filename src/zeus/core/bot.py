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
from ..exchanges.base import OrderSide, OrderType, OrderStatus, OHLCV
from ..indicators.prebreakout_detector import PreBreakoutDetector
from ..strategies.signal_generator import SignalGenerator, TradingSignal, StrategyMode
from ..strategies.risk_manager import RiskManager, RiskConfig
from ..alerts.telegram_bot import TelegramAlerts, AlertConfig
from ..ml.learning_engine import TradingLearningEngine, TradeOutcome
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
        alert_config = AlertConfig(enabled=bool(telegram_token and telegram_chat_id))
        alert_config.enforce_trade_only_alerts()
        self.telegram = TelegramAlerts(
            telegram_token, 
            telegram_chat_id,
            alert_config
        )
        self.ml_engine = TradingLearningEngine()
        self.mode = mode
        self.running = False
        self._pairs_cache: List[str] = []
        self._last_pair_refresh = 0
        self._learning_cycle_counter = 0

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
            await self._sync_holdings_with_positions(holdings)
            self.risk_manager.update_portfolio(equity, [])
            logger.info(f"Balance refreshed: ${equity:.2f} | Holdings: {len(holdings)} tokens")
        except Exception as e:
            logger.error(f"Failed to refresh balance: {e}")

    async def _sync_holdings_with_positions(self, holdings: Dict[str, float]) -> None:
        """Ensure all token holdings are tracked - no unaccounted tokens"""
        tracked_symbols = self.state.get_active_symbols()
        for token, amount in holdings.items():
            if token in ["USD", "ZUSD", "EUR", "ZEUR"]:
                continue
            if amount < 0.00001:
                continue
            pair = f"{token}USD"
            if pair not in tracked_symbols and token not in tracked_symbols:
                try:
                    ticker = await self.exchange.fetch_ticker(pair)
                    if ticker and ticker.last > 0:
                        logger.info(f"Found untracked holding: {token} = {amount}, creating position record")
                        trade = TradeRecord(
                            id=f"SYNC_{token}_{int(time.time())}",
                            symbol=pair,
                            side="buy",
                            entry_price=ticker.last,
                            size=amount,
                            stop_loss=ticker.last * 0.95,
                            take_profit=ticker.last * 1.10,
                            status="open",
                            entry_time=datetime.now(timezone.utc).isoformat(),
                            strategy="synced",
                            confidence=50.0,
                            prebreakout_score=0.0,
                            is_manual=True,
                            protected=True
                        )
                        await self.state.open_trade(trade)
                except Exception as e:
                    logger.debug(f"Could not sync {token}: {e}")

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
                    ohlcv = await self.exchange.fetch_ohlcv(pair, "1h", 500)
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
                ohlcv = await self.exchange.fetch_ohlcv(candidate["symbol"], "15m", 500)
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
                order_book = await self.exchange.analyze_order_book(signal.symbol)
                optimal_price = signal.entry_price
                if order_book and signal.side == OrderSide.BUY:
                    best_bid = order_book.get("best_bid", signal.entry_price)
                    spread_pct = order_book.get("spread_pct", 0)
                    if spread_pct > 0.5:
                        optimal_price = best_bid * 0.9995
                    else:
                        optimal_price = best_bid * 0.9998
                    optimal_price = max(optimal_price, signal.entry_price * 0.995)
                    logger.info(f"Optimized buy price: ${optimal_price:.8f} (bid: ${best_bid:.8f})")
                elif order_book and signal.side == OrderSide.SELL:
                    best_ask = order_book.get("best_ask", signal.entry_price)
                    optimal_price = best_ask * 1.0002
                    logger.info(f"Optimized sell price: ${optimal_price:.8f} (ask: ${best_ask:.8f})")
                order = await self.exchange.create_order(
                    signal.symbol,
                    OrderType.LIMIT,
                    signal.side,
                    size,
                    price=optimal_price
                )
                used_market_order = False
                if not order:
                    logger.warning("Limit order failed, falling back to market order")
                    order = await self.exchange.create_order(
                        signal.symbol,
                        OrderType.MARKET,
                        signal.side,
                        size
                    )
                    used_market_order = True
                if not order:
                    logger.error("Order placement failed")
                    return None
                trade_id = order.id
                if not used_market_order:
                    filled = False
                    for _ in range(10):
                        await asyncio.sleep(1)
                        order_status = await self.exchange.fetch_order(trade_id, signal.symbol)
                        if order_status and order_status.status == OrderStatus.FILLED:
                            filled = True
                            logger.info(f"Limit order filled: {trade_id} @ ${optimal_price:.8f}")
                            break
                    if not filled:
                        logger.warning(f"Limit order not filled within 10s, canceling and using market")
                        await self.exchange.cancel_order(trade_id, signal.symbol)
                        order = await self.exchange.create_order(
                            signal.symbol,
                            OrderType.MARKET,
                            signal.side,
                            size
                        )
                        if not order:
                            logger.error("Market order fallback failed")
                            return None
                        trade_id = order.id
                        used_market_order = True
                if used_market_order:
                    logger.info(f"Market order executed: {trade_id}")
                else:
                    logger.info(f"Limit order filled: {trade_id} @ ${optimal_price:.8f}")
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
                
                if trade.price_history is None:
                    trade.price_history = []
                trade.price_history.append(current_price)
                if len(trade.price_history) > 60:
                    trade.price_history = trade.price_history[-60:]
                
                if trade.side == "buy":
                    if trade.peak_price == 0:
                        trade.peak_price = trade.entry_price
                    
                    if current_price > trade.peak_price:
                        old_peak = trade.peak_price
                        trade.peak_price = current_price
                        trade.peak_count += 1
                        
                        peak_gain = (current_price - old_peak) / old_peak if old_peak > 0 else 0
                        if peak_gain < 0.005:
                            trade.fakeout_signals += 1
                        
                        if not trade.breakout_confirmed and current_price > trade.entry_price * 1.03:
                            trade.breakout_confirmed = True
                            trade.breakout_strength = (current_price - trade.entry_price) / trade.entry_price
                            logger.info(f"{trade.symbol} BREAKOUT CONFIRMED at ${current_price:.6f} (+{trade.breakout_strength:.1%})")
                        
                        logger.info(f"{trade.symbol} new peak: ${current_price:.6f} (peak #{trade.peak_count})")
                    
                    profit_from_entry = (current_price - trade.entry_price) / trade.entry_price
                    drop_from_peak = (trade.peak_price - current_price) / trade.peak_price if trade.peak_price > 0 else 0
                    
                    if trade.breakout_confirmed:
                        trailing_drop_pct = 0.015
                        min_profit_pct = 0.03
                    else:
                        trailing_drop_pct = 0.025
                        min_profit_pct = 0.04
                    
                    if profit_from_entry >= min_profit_pct and drop_from_peak >= trailing_drop_pct:
                        should_close = True
                        close_reason = f"Trailing Peak Sell (peak ${trade.peak_price:.6f}, dropped {drop_from_peak:.1%})"
                    
                    if trade.fakeout_signals >= 5 and profit_from_entry > 0.02:
                        should_close = True
                        close_reason = f"Fakeout Protection ({trade.fakeout_signals} weak peaks detected)"
                    
                    if len(trade.price_history) >= 10:
                        recent_prices = trade.price_history[-10:]
                        declining_count = sum(1 for i in range(1, len(recent_prices)) if recent_prices[i] < recent_prices[i-1])
                        if declining_count >= 8 and profit_from_entry > 0.015:
                            should_close = True
                            close_reason = "Momentum Reversal (8+ declining ticks)"
                    
                    if trade.breakout_confirmed and drop_from_peak >= 0.03:
                        should_close = True
                        close_reason = f"Breakout Reversal Protection (dropped {drop_from_peak:.1%} from peak)"
                    
                    if current_price >= trade.take_profit:
                        should_close = True
                        close_reason = "Take Profit Hit"
                else:
                    if trade.peak_price == 0:
                        trade.peak_price = trade.entry_price
                    
                    if current_price < trade.peak_price:
                        trade.peak_price = current_price
                        trade.peak_count += 1
                    
                    profit_from_entry = (trade.entry_price - current_price) / trade.entry_price
                    rise_from_peak = (current_price - trade.peak_price) / trade.peak_price if trade.peak_price > 0 else 0
                    
                    if profit_from_entry >= 0.015 and rise_from_peak >= 0.015:
                        should_close = True
                        close_reason = f"Trailing Peak Sell (peak ${trade.peak_price:.6f})"
                    
                    if current_price <= trade.take_profit:
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
        pnl = (exit_price - trade.entry_price) * trade.size if trade.side == "buy" else (trade.entry_price - exit_price) * trade.size
        pnl_pct = (exit_price - trade.entry_price) / trade.entry_price * 100 if trade.entry_price > 0 else 0
        if trade.side == "sell":
            pnl_pct = -pnl_pct
        is_emergency_stop = "Emergency" in reason or "Stop Loss" in reason
        if trade.is_manual or trade.protected:
            if pnl_pct < 5.0 and not is_emergency_stop:
                logger.info(f"Protected trade {trade.symbol} - waiting for 5%+ profit (current: {pnl_pct:.2f}%)")
                return
        if pnl < 0 and not is_emergency_stop:
            entry_time = datetime.fromisoformat(trade.entry_time.replace('Z', '+00:00'))
            hold_time = (datetime.now(timezone.utc) - entry_time).total_seconds()
            max_hold_time = 7 * 24 * 3600
            if hold_time < max_hold_time:
                logger.info(f"Loss prevention: {trade.symbol} at {pnl_pct:.2f}% - holding for recovery")
                return
            else:
                logger.warning(f"Max hold time exceeded for {trade.symbol} - allowing loss exit after {hold_time/3600:.1f}h")
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
            pnl_pct = (exit_price - closed_trade.entry_price) / closed_trade.entry_price * 100
            entry_time = datetime.fromisoformat(closed_trade.entry_time.replace('Z', '+00:00'))
            exit_time = datetime.now(timezone.utc)
            duration = int((exit_time - entry_time).total_seconds())
            indicators_snapshot = {}
            try:
                ohlcv = await self.exchange.fetch_ohlcv(closed_trade.symbol, "1h", 500)
                if len(ohlcv) >= 20:
                    close_prices = [c.close for c in ohlcv]
                    indicators_snapshot = {
                        "last_price": close_prices[-1],
                        "price_change_24h": (close_prices[-1] - close_prices[-24]) / close_prices[-24] * 100 if len(close_prices) >= 24 else 0
                    }
            except:
                pass
            outcome = TradeOutcome(
                symbol=closed_trade.symbol,
                side=closed_trade.side,
                entry_price=closed_trade.entry_price,
                exit_price=exit_price,
                size=closed_trade.size,
                pnl=closed_trade.pnl,
                pnl_pct=pnl_pct,
                duration_seconds=duration,
                strategy=closed_trade.strategy,
                confidence=closed_trade.confidence,
                prebreakout_score=closed_trade.prebreakout_score,
                entry_time=closed_trade.entry_time,
                exit_time=exit_time.isoformat(),
                exit_reason=reason,
                indicators_at_entry=indicators_snapshot
            )
            self.ml_engine.record_trade_outcome(outcome)
            logger.info(f"Trade closed: {closed_trade.symbol} PnL: ${closed_trade.pnl:.2f} ({pnl_pct:+.2f}%)")

    async def run_cycle(self) -> None:
        try:
            await self._refresh_balance()
            candidates = await self.scan_markets()
            if candidates:
                signals = await self.generate_signals(candidates)
                for signal in signals[:5]:
                    if self.state.can_open_new_trade():
                        should_trade, adjusted_conf, ml_reasons = self.ml_engine.get_trade_recommendation(
                            signal.symbol,
                            signal.confidence,
                            signal.prebreakout_score,
                            signal.strategy_mode.value
                        )
                        if should_trade and adjusted_conf >= self.state.state.config.min_confidence:
                            signal.confidence = adjusted_conf
                            if ml_reasons:
                                signal.reasons.extend(ml_reasons)
                            await self.execute_trade(signal)
                        else:
                            logger.info(f"ML blocked trade for {signal.symbol}: {ml_reasons}")
                    else:
                        logger.info(f"Max positions ({self.state.state.config.max_open_positions}) reached, monitoring existing trades")
            await self.manage_positions()
            self._learning_cycle_counter += 1
            if self._learning_cycle_counter >= 100:
                self._learning_cycle_counter = 0
                result = self.ml_engine.run_learning_cycle()
                logger.info(f"ML Learning cycle: {result}")
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
