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
import pytz

LA_TZ = pytz.timezone('America/Los_Angeles')

from ..exchanges.kraken import KrakenExchange
from ..exchanges.base import OrderSide, OrderType, OrderStatus, OHLCV
from ..indicators.prebreakout_detector import PreBreakoutDetector
from ..strategies.signal_generator import SignalGenerator, TradingSignal, StrategyMode
from ..strategies.risk_manager import RiskManager, RiskConfig
from ..alerts.telegram_bot import TelegramAlerts, AlertConfig
from ..ml.learning_engine import TradingLearningEngine, TradeOutcome
from ..risk.advanced_risk import AdvancedRiskEngine
from ..data.alternative_data import AlternativeDataAggregator
from ..execution.smart_execution import SmartExecutionEngine
from ..indicators.microstructure import MicrostructureAnalyzer
from ..analytics.kpi_tracker import KPITracker, TradeMetrics
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
        self.advanced_risk = AdvancedRiskEngine()
        self.alt_data = AlternativeDataAggregator()
        self.smart_exec = SmartExecutionEngine()
        self.microstructure = MicrostructureAnalyzer()
        self.kpi_tracker = KPITracker()
        self.mode = mode
        self.running = False
        self._pairs_cache: List[str] = []
        self._last_pair_refresh = 0
        self._learning_cycle_counter = 0
        self._last_full_scan = 0
        self._last_position_scan = 0
        self._all_timeframes = ["5m", "15m", "1h", "4h", "1d"]
        self._priority_candidates: List[Dict[str, Any]] = []
        self._priority_focus_size = 20
        self._priority_last_refresh = 0
        self._priority_refresh_interval = 900
        self._last_priority_scan = 0
        self._last_alt_data_update = 0
        self._last_equity_snapshot = 0.0
        self._last_equity_date = ""

    async def start(self) -> bool:
        logger.info(f"Starting Zeus Bot in {self.mode} mode...")
        if not await self.exchange.connect():
            logger.error("Failed to connect to Kraken")
            return False
        self.advanced_risk.reset_circuit_breaker()
        self.advanced_risk.reset_daily_stats()
        await self.telegram.initialize()
        self.telegram.set_bot_reference(self)
        await self.telegram.start_command_listener()
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
        await self.telegram.stop_command_listener()
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
            self._record_daily_return_if_new_day(current_equity)
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
            self._record_daily_return_if_new_day(equity)
            logger.info(f"Balance refreshed: ${equity:.2f} | Holdings: {len(holdings)} tokens")
        except Exception as e:
            logger.error(f"Failed to refresh balance: {e}")

    def _record_daily_return_if_new_day(self, current_equity: float) -> None:
        """Record daily return for KPI tracking when day changes"""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._last_equity_date != today and self._last_equity_snapshot > 0:
            daily_return_pct = ((current_equity - self._last_equity_snapshot) / self._last_equity_snapshot) * 100
            self.kpi_tracker.record_daily_return(daily_return_pct)
            self.kpi_tracker.update_equity(current_equity)
            logger.info(f"[KPI] Daily return recorded: {daily_return_pct:+.2f}% ({self._last_equity_date})")
        if self._last_equity_date != today:
            self._last_equity_snapshot = current_equity
            self._last_equity_date = today

    async def _sync_holdings_with_positions(self, holdings: Dict[str, float]) -> None:
        """Ensure all token holdings are tracked - no unaccounted tokens"""
        tracked_symbols = self.state.get_active_symbols()
        
        already_synced = set()
        for trade in self.state.get_open_trades():
            if trade.synced or trade.strategy == "synced":
                token = trade.symbol.replace("USD", "").replace("USDT", "").replace("USDC", "")
                already_synced.add(token)
        
        for token, amount in holdings.items():
            if token in ["USD", "ZUSD", "EUR", "ZEUR", "USDC", "USDT"]:
                continue
            if amount < 0.00001:
                continue
            if token in already_synced:
                continue
            pair = f"{token}USD"
            if pair not in tracked_symbols and token not in tracked_symbols:
                try:
                    await asyncio.sleep(0.3)
                    ticker = await self.exchange.fetch_ticker(pair)
                    if ticker and ticker.last > 0:
                        logger.info(f"Found untracked holding: {token} = {amount}, creating position record")
                        take_profit, stop_loss = await self._calculate_intelligent_targets(pair, ticker.last, amount)
                        trade = TradeRecord(
                            id=f"SYNC_{token}_{int(time.time())}",
                            symbol=pair,
                            side="buy",
                            entry_price=ticker.last,
                            size=amount,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            status="open",
                            entry_time=datetime.now(timezone.utc).isoformat(),
                            strategy="synced",
                            confidence=50.0,
                            prebreakout_score=0.0,
                            is_manual=True,
                            protected=True,
                            synced=True
                        )
                        sell_order_id = await self._place_limit_sell(pair, amount, take_profit)
                        if sell_order_id:
                            trade.sell_order_id = sell_order_id
                            await self.state.open_trade(trade)
                            logger.info(f"Synced {token} with sell order {sell_order_id}")
                        else:
                            trade.sell_order_attempts += 1
                            if trade.sell_order_attempts < 3:
                                await self.state.open_trade(trade)
                                logger.info(f"Synced {token} without sell order (will retry later)")
                            else:
                                logger.warning(f"Could not place sell order for {token}, skipping (tokens may be locked or staked)")
                except Exception as e:
                    logger.debug(f"Could not sync {token}: {e}")

    async def _calculate_intelligent_targets(self, symbol: str, current_price: float, size: float) -> tuple:
        """Calculate intelligent take-profit and stop-loss using KPIs, volatility, and market analysis"""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, "1h", 100)
            if len(ohlcv) < 50:
                return current_price * 1.08, current_price * 0.92
            high = [c.high for c in ohlcv]
            low = [c.low for c in ohlcv]
            close = [c.close for c in ohlcv]
            volume = [c.volume for c in ohlcv]
            indicators = self.signal_gen.math.calculate_all(high, low, close, volume)
            atr = indicators.get("atr", current_price * 0.03)
            recent_high = max(high[-20:])
            recent_low = min(low[-20:])
            price_range = recent_high - recent_low
            volatility_pct = (atr / current_price) * 100 if current_price > 0 else 3.0
            resistance_levels = []
            for i in range(len(high) - 5):
                if high[i] == max(high[max(0, i-2):i+3]):
                    resistance_levels.append(high[i])
            resistance_levels = sorted([r for r in resistance_levels if r > current_price])[:3]
            rsi = indicators.get("rsi", 50)
            macd_data = indicators.get("macd", {})
            macd_hist = macd_data.get("histogram", 0) if isinstance(macd_data, dict) else 0
            bullish_momentum = rsi < 70 and macd_hist > 0
            if volatility_pct > 8:
                tp_mult = 2.5
            elif volatility_pct > 5:
                tp_mult = 2.2
            elif volatility_pct > 3:
                tp_mult = 2.0
            else:
                tp_mult = 1.8
            if bullish_momentum:
                tp_mult *= 1.15
            base_tp = current_price + (atr * tp_mult)
            if resistance_levels:
                nearest_resistance = resistance_levels[0]
                if nearest_resistance < base_tp * 1.2:
                    base_tp = nearest_resistance * 0.995
            take_profit = max(base_tp, current_price * 1.04)
            sl_mult = 1.5
            stop_loss = current_price - (atr * sl_mult)
            stop_loss = max(stop_loss, current_price * 0.88)
            logger.info(f"{symbol} intelligent targets: TP=${take_profit:.6f} (+{((take_profit/current_price)-1)*100:.1f}%), SL=${stop_loss:.6f} (-{((1-stop_loss/current_price))*100:.1f}%) | ATR={atr:.6f} Vol={volatility_pct:.1f}%")
            return round(take_profit, 8), round(stop_loss, 8)
        except Exception as e:
            logger.warning(f"Could not calculate intelligent targets for {symbol}: {e}")
            return current_price * 1.08, current_price * 0.92

    def _round_price_for_kraken(self, symbol: str, price: float) -> float:
        """Round price to appropriate decimal places for Kraken based on price magnitude
        
        Kraken has strict precision requirements per pair. Most pairs allow fewer decimals
        than the price magnitude suggests. We use conservative rounding to avoid rejection.
        """
        if price >= 10000:
            return round(price, 1)
        elif price >= 1000:
            return round(price, 2)
        elif price >= 100:
            return round(price, 2)
        elif price >= 10:
            return round(price, 3)
        elif price >= 1:
            return round(price, 3)
        elif price >= 0.1:
            return round(price, 4)
        elif price >= 0.01:
            return round(price, 4)
        elif price >= 0.001:
            return round(price, 5)
        elif price >= 0.0001:
            return round(price, 6)
        else:
            return round(price, 6)

    async def _place_limit_sell(self, symbol: str, size: float, price: float) -> Optional[str]:
        """Place a limit sell order on the exchange"""
        if self.mode != "LIVE":
            logger.info(f"PAPER mode: Would place sell order for {symbol} @ ${price:.6f}")
            return None
        price = self._round_price_for_kraken(symbol, price)
        try:
            sell_order = await self.exchange.create_order(
                symbol,
                OrderType.LIMIT,
                OrderSide.SELL,
                size,
                price=price
            )
            if sell_order and sell_order.id:
                logger.info(f"Placed limit sell order for {symbol}: {sell_order.id} @ ${price:.6f}")
                return sell_order.id
        except Exception as e:
            logger.warning(f"Failed to place sell order for {symbol}: {e}")
        return None

    async def ensure_sell_orders_placed(self) -> None:
        """Check all bot-managed positions and ensure they have sell orders on exchange"""
        if self.mode != "LIVE":
            return
        holdings = self.state.state.holdings
        logger.debug(f"Current holdings: {holdings}")
        positions_to_remove = []
        try:
            all_open_orders = await self.exchange.fetch_open_orders()
            logger.info(f"Fetched {len(all_open_orders)} open orders from Kraken")
            for o in all_open_orders[:5]:
                logger.debug(f"Order: {o.symbol} {o.side.value} {o.amount} @ {o.price}")
        except Exception as e:
            logger.warning(f"Failed to fetch open orders: {e}")
            all_open_orders = []
        orders_by_symbol = {}
        for order in all_open_orders:
            if order.side.value == "sell" or str(order.side) == "OrderSide.SELL":
                sym = order.symbol
                if sym not in orders_by_symbol:
                    orders_by_symbol[sym] = []
                orders_by_symbol[sym].append(order)
        if orders_by_symbol:
            logger.info(f"Found existing sell orders for: {list(orders_by_symbol.keys())}")
        for trade_id, trade in list(self.state.state.active_trades.items()):
            if trade.protected or trade.is_manual:
                continue
            if trade.symbol in ["USDCUSD", "USDTUSD"]:
                continue
            token = trade.symbol.replace("USD", "")
            token_balance = holdings.get(token, 0)
            if token_balance < 0.00001:
                logger.warning(f"Position {trade.symbol} has no tokens (balance: {token_balance:.8f}), marking for removal")
                positions_to_remove.append(trade_id)
                continue
            if token_balance < trade.size * 0.5:
                logger.info(f"Position {trade.symbol} balance {token_balance:.4f} < size {trade.size:.4f}, adjusting size")
                trade.size = token_balance
            existing_sell_orders = orders_by_symbol.get(trade.symbol, [])
            if existing_sell_orders and not trade.sell_order_id:
                trade.sell_order_id = existing_sell_orders[0].id
                trade.take_profit = existing_sell_orders[0].price
                logger.info(f"Synced existing sell order for {trade.symbol}: {trade.sell_order_id}")
                await self.state.save_state()
                continue
            if not trade.sell_order_id and not existing_sell_orders:
                logger.info(f"Position {trade.symbol} missing sell order, placing now...")
                take_profit, _ = await self._calculate_intelligent_targets(
                    trade.symbol, trade.entry_price, trade.size
                )
                trade.take_profit = take_profit
                sell_order_id = await self._place_limit_sell(trade.symbol, token_balance, take_profit)
                if sell_order_id:
                    trade.sell_order_id = sell_order_id
                    trade.size = token_balance
                    await self.state.save_state()
                    logger.info(f"Placed missing sell order for {trade.symbol}: {sell_order_id}")
                else:
                    trade.sell_order_attempts = getattr(trade, 'sell_order_attempts', 0) + 1
                    if trade.sell_order_attempts >= 3:
                        logger.warning(f"Position {trade.symbol} failed to place sell order 3 times, marking for removal (tokens may be locked)")
                        positions_to_remove.append(trade_id)
        for trade_id in positions_to_remove:
            if trade_id in self.state.state.active_trades:
                del self.state.state.active_trades[trade_id]
                logger.info(f"Removed ghost position: {trade_id}")
        if positions_to_remove:
            await self.state.save_state()

    async def _refresh_pairs(self) -> None:
        try:
            self._pairs_cache = await self.exchange.fetch_usd_pairs()
            self._last_pair_refresh = time.time()
            logger.info(f"Refreshed pairs list: {len(self._pairs_cache)} USD pairs")
        except Exception as e:
            logger.error(f"Failed to refresh pairs: {e}")

    async def _analyze_multi_timeframe(self, pair: str) -> Optional[Dict[str, Any]]:
        try:
            mtf_data = {}
            mtf_scores = []
            for tf in self._all_timeframes:
                await asyncio.sleep(0.1)
                ohlcv = await self.exchange.fetch_ohlcv(pair, tf, 500)
                if len(ohlcv) < 50:
                    continue
                high = [c.high for c in ohlcv]
                low = [c.low for c in ohlcv]
                close = [c.close for c in ohlcv]
                volume = [c.volume for c in ohlcv]
                analysis = await self.prebreakout.analyze(pair, high, low, close, volume)
                mtf_data[tf] = analysis
                if analysis.get("prebreakout_score", 0) > 0:
                    weight = {"5m": 0.1, "15m": 0.15, "1h": 0.25, "4h": 0.30, "1d": 0.20}.get(tf, 0.15)
                    mtf_scores.append(analysis["prebreakout_score"] * weight)
            if not mtf_scores:
                return None
            combined_score = sum(mtf_scores) / sum([0.1, 0.15, 0.25, 0.30, 0.20][:len(mtf_scores)])
            primary_tf = mtf_data.get("1h", mtf_data.get("4h", mtf_data.get("15m", {})))
            if not primary_tf:
                return None
            bullish_count = sum(1 for tf, d in mtf_data.items() if d.get("stage") in ["PRE_BREAKOUT", "BREAKOUT"])
            if bullish_count >= 2 or combined_score >= 55:
                order_book = await self.exchange.analyze_order_book(pair)
                return {
                    "symbol": pair,
                    "price": primary_tf.get("close_price", 0),
                    "prebreakout_score": combined_score,
                    "stage": primary_tf.get("stage", "UNKNOWN"),
                    "buy_anchor": primary_tf.get("buy_anchor", 0),
                    "sell_anchor": primary_tf.get("sell_anchor", 0),
                    "stop_loss": primary_tf.get("stop_loss", 0),
                    "take_profit": primary_tf.get("take_profit", 0),
                    "atr": primary_tf.get("atr", 0),
                    "features": primary_tf.get("features", {}),
                    "reasons": [f"MTF aligned ({bullish_count}/5 bullish)"] + primary_tf.get("reasons", []),
                    "spread_pct": order_book.get("spread_pct", 0) if order_book else 0,
                    "imbalance": order_book.get("imbalance", 0) if order_book else 0,
                    "mtf_alignment": bullish_count / len(mtf_data) if mtf_data else 0,
                    "kpi_count": primary_tf.get("kpi_count", 23)
                }
        except Exception as e:
            logger.debug(f"MTF analysis error for {pair}: {e}")
        return None

    async def scan_markets(self) -> List[Dict[str, Any]]:
        await self.state.set_status("SCANNING")
        logger.info(f"Full market scan: {len(self._pairs_cache)} pairs (all timeframes)...")
        if time.time() - self._last_alt_data_update > 1800:
            try:
                fg_data = await self.alt_data.update_fear_greed()
                if fg_data:
                    logger.info(f"Fear & Greed: {fg_data.value} ({fg_data.classification})")
                self._last_alt_data_update = time.time()
            except Exception as e:
                logger.debug(f"Alt data update error: {e}")
        portfolio_value = self.state.state.equity
        positions = [
            {"symbol": t.symbol, "size": t.size, "entry_price": t.entry_price, 
             "current_price": t.entry_price}
            for t in self.state.get_open_trades()
        ]
        if portfolio_value > 0:
            port_risk = self.advanced_risk.assess_portfolio_risk(positions, portfolio_value)
            if port_risk.circuit_breaker_active:
                logger.warning("CIRCUIT BREAKER: Trading halted due to risk limits")
                self._last_full_scan = time.time()
                await self.state.set_status("PAUSED_CIRCUIT_BREAKER")
                return []
        if time.time() - self._last_pair_refresh > 3600:
            await self._refresh_pairs()
        candidates = []
        semaphore = asyncio.Semaphore(5)
        async def analyze_pair(pair: str) -> Optional[Dict[str, Any]]:
            async with semaphore:
                return await self._analyze_multi_timeframe(pair)
        results = await asyncio.gather(*[analyze_pair(p) for p in self._pairs_cache])
        candidates = [r for r in results if r is not None]
        candidates.sort(key=lambda x: x["prebreakout_score"], reverse=True)
        self._priority_candidates = candidates[:self._priority_focus_size]
        self._priority_last_refresh = time.time()
        logger.info(f"Top-{self._priority_focus_size} priority candidates identified:")
        for i, c in enumerate(self._priority_candidates[:10]):
            logger.info(f"  #{i+1}: {c['symbol']} Score: {c['prebreakout_score']:.1f} Stage: {c['stage']} KPIs: {c.get('kpi_count', 23)}")
        await self.state.set_candidates(candidates[:50])
        await self.state.set_status("IDLE")
        self._last_full_scan = time.time()
        logger.info(f"Full scan complete. Found {len(candidates)} candidates. Focusing on top {len(self._priority_candidates)}.")
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

    async def scan_priority_candidates(self) -> List[Dict[str, Any]]:
        if not self._priority_candidates:
            return await self.scan_markets()
        if time.time() - self._priority_last_refresh > self._priority_refresh_interval:
            logger.info("Priority candidates expired, running full market refresh...")
            return await self.scan_markets()
        logger.info(f"Priority scan: Re-analyzing top {len(self._priority_candidates)} candidates...")
        refreshed_candidates = []
        semaphore = asyncio.Semaphore(5)
        active_symbols = set(self.state.get_active_symbols())
        async def analyze_priority(candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            async with semaphore:
                if candidate["symbol"] in active_symbols:
                    return candidate
                return await self._analyze_multi_timeframe(candidate["symbol"])
        results = await asyncio.gather(*[analyze_priority(c) for c in self._priority_candidates])
        refreshed_candidates = [r for r in results if r is not None]
        refreshed_candidates.sort(key=lambda x: x.get("prebreakout_score", 0), reverse=True)
        exhausted = [c for c in refreshed_candidates if c.get("stage") in ["NEUTRAL", "DECLINE", "DISTRIBUTION"] or c.get("prebreakout_score", 0) < 40]
        if len(exhausted) > len(self._priority_candidates) * 0.5:
            logger.info(f"More than half of priority candidates exhausted ({len(exhausted)}/{len(self._priority_candidates)}), refreshing market...")
            return await self.scan_markets()
        self._priority_candidates = [c for c in refreshed_candidates if c.get("prebreakout_score", 0) >= 35][:self._priority_focus_size]
        logger.info(f"Priority scan complete. {len(self._priority_candidates)} candidates still active.")
        return refreshed_candidates

    async def scan_active_positions(self) -> None:
        open_trades = self.state.get_open_trades()
        if not open_trades:
            return
        bot_trades = [t for t in open_trades if not t.is_manual and not t.protected]
        if bot_trades:
            logger.info(f"Position scan: {len(bot_trades)} bot-managed positions (all timeframes)...")
            for trade in bot_trades:
                try:
                    await asyncio.sleep(0.2)
                    mtf_analysis = await self._analyze_multi_timeframe(trade.symbol)
                    if mtf_analysis:
                        trade.mtf_alignment = mtf_analysis.get("mtf_alignment", 0)
                        if mtf_analysis.get("stage") in ["DISTRIBUTION", "DECLINE"] and mtf_analysis.get("prebreakout_score", 100) < 40:
                            logger.warning(f"{trade.symbol} MTF signals weakening (score: {mtf_analysis.get('prebreakout_score', 0):.1f})")
                except Exception as e:
                    logger.debug(f"Position scan error for {trade.symbol}: {e}")
        self._last_position_scan = time.time()
        logger.info(f"Position scan complete.")

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
        logger.info(f"[TRADE GATE] Attempting trade: {signal.symbol} conf={signal.confidence:.1f} score={signal.prebreakout_score:.1f}")
        if not self.state.can_open_new_trade():
            current_count = len(self.state.state.active_trades)
            max_positions = self.state.state.config.max_open_positions
            logger.warning(f"[TRADE BLOCKED] Max positions reached ({current_count}/{max_positions}), skipping {signal.symbol}")
            return None
        if signal.symbol in self.state.get_active_symbols():
            logger.warning(f"[TRADE BLOCKED] Already have position in {signal.symbol}")
            return None
        can_trade, reasons = self.risk_manager.can_open_position(signal.symbol)
        if not can_trade:
            logger.warning(f"[TRADE BLOCKED] Risk check failed for {signal.symbol}: {reasons}")
            return None
        current_positions = [
            {"symbol": t.symbol, "size": t.size, "entry_price": t.entry_price, 
             "current_price": t.entry_price}
            for t in self.state.get_open_trades()
        ]
        portfolio_value = self.state.state.equity
        adv_risk = self.advanced_risk.assess_trade_risk(
            signal.symbol, 1.0, signal.entry_price, signal.stop_loss,
            signal.take_profit, portfolio_value, current_positions
        )
        if not adv_risk.approved:
            logger.warning(f"[TRADE BLOCKED] Advanced risk rejected for {signal.symbol}: {adv_risk.rejection_reasons}")
            return None
        if adv_risk.warnings:
            logger.info(f"[TRADE INFO] Risk warnings for {signal.symbol}: {adv_risk.warnings}")
        size = self.risk_manager.calculate_optimal_size(
            signal.entry_price,
            signal.stop_loss,
            signal.confidence,
            signal.take_profit
        )
        if size <= 0:
            logger.warning(f"[TRADE BLOCKED] Position size too small for {signal.symbol} at price ${signal.entry_price:.6f}")
            return None
        if adv_risk.kelly_size > 0:
            size = min(size, adv_risk.kelly_size)
        projected_profit_pct = ((signal.take_profit - signal.entry_price) / signal.entry_price) * 100 if signal.take_profit > signal.entry_price else 0
        if projected_profit_pct >= 100:
            profit_cap_mult = 10.0
        elif projected_profit_pct >= 50:
            profit_cap_mult = 5.0
        elif projected_profit_pct >= 30:
            profit_cap_mult = 3.0
        elif projected_profit_pct >= 15:
            profit_cap_mult = 2.0
        else:
            profit_cap_mult = 1.0
        max_trade_amount = self.state.state.config.per_trade_amount * profit_cap_mult
        max_trade_amount = min(max_trade_amount, self.state.state.equity * 0.25)
        max_size = max_trade_amount / signal.entry_price
        size = min(size, max_size)
        if profit_cap_mult > 1.0:
            logger.info(f"Profit-scaled sizing: {projected_profit_pct:.1f}% projected → {profit_cap_mult}x cap (max ${max_trade_amount:.2f})")
        
        order_value = size * signal.entry_price
        min_order_value = 5.0
        if order_value < min_order_value:
            logger.warning(f"[TRADE BLOCKED] Order value ${order_value:.2f} below minimum ${min_order_value:.2f}, skipping {signal.symbol}")
            return None
        logger.info(f"[TRADE APPROVED] {signal.symbol}: size={size:.6f} value=${order_value:.2f} conf={signal.confidence:.1f}%")
        trade_id = f"T{int(time.time() * 1000)}"
        if self.mode == "LIVE":
            try:
                order_book = await self.exchange.analyze_order_book(signal.symbol)
                volatility_pct = signal.atr / signal.entry_price if signal.atr > 0 else 0.02
                from ..execution.smart_execution import MarketConditions
                market_conditions = MarketConditions(
                    spread=order_book.get("spread", 0.01) if order_book else 0.01,
                    spread_pct=order_book.get("spread_pct", 0.5) if order_book else 0.5,
                    bid_depth=order_book.get("total_bid_volume", 1000) if order_book else 1000,
                    ask_depth=order_book.get("total_ask_volume", 1000) if order_book else 1000,
                    volatility=volatility_pct,
                    volume_24h=order_book.get("volume_24h", 10000) if order_book else 10000,
                    recent_volume=order_book.get("recent_volume", 1000) if order_book else 1000,
                    liquidity_score=0.5
                )
                exec_strategy = self.smart_exec.select_strategy(
                    order_size=size,
                    order_value=order_value,
                    market_conditions=market_conditions,
                    urgency=signal.confidence / 100.0
                )
                logger.info(f"Smart execution strategy: {exec_strategy}")
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
        la_time = datetime.now(LA_TZ)
        sell_order_id = None
        if signal.side == OrderSide.BUY and self.mode == "LIVE":
            try:
                sell_order = await self.exchange.create_order(
                    signal.symbol,
                    OrderType.LIMIT,
                    OrderSide.SELL,
                    size,
                    price=signal.take_profit
                )
                if sell_order:
                    sell_order_id = sell_order.id
                    logger.info(f"Immediate sell order placed: {sell_order_id} @ ${signal.take_profit:.8f} (TP)")
                else:
                    logger.warning(f"Failed to place immediate sell order for {signal.symbol}")
            except Exception as e:
                logger.warning(f"Could not place immediate sell order: {e}")
        trade = TradeRecord(
            id=trade_id,
            symbol=signal.symbol,
            side=signal.side.value,
            entry_price=signal.entry_price,
            size=size,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            status="open",
            entry_time=la_time.isoformat(),
            strategy=signal.strategy_mode.value,
            confidence=signal.confidence,
            prebreakout_score=signal.prebreakout_score
        )
        if sell_order_id:
            trade.sell_order_id = sell_order_id
        await self.state.open_trade(trade)
        await self.telegram.send_trade_opened(
            signal.symbol,
            signal.side,
            signal.entry_price,
            size,
            signal.stop_loss,
            signal.take_profit
        )
        logger.info(f"Trade opened: {signal.symbol} {signal.side.value} @ {signal.entry_price} ({la_time.strftime('%I:%M:%S %p PT')})")
        return trade_id

    async def manage_positions(self) -> None:
        for trade_id, trade in list(self.state.state.active_trades.items()):
            if trade.is_manual or trade.protected:
                continue
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
                
                if trade.sell_order_id and self.mode == "LIVE":
                    try:
                        order_status = await self.exchange.fetch_order(trade.sell_order_id, trade.symbol)
                        if order_status and order_status.status == OrderStatus.FILLED:
                            trade.sell_order_id = None
                            logger.info(f"{trade.symbol} sell order filled at TP - closing position")
                            await self._close_position(trade_id, trade.take_profit, "Take Profit Order Filled")
                            continue
                    except Exception as e:
                        logger.debug(f"Sell order check error: {e}")
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
                        if trade.sell_order_id and self.mode == "LIVE" and trade.peak_count % 5 == 0:
                            try:
                                new_tp = current_price * 1.035
                                if new_tp > trade.take_profit * 1.05:
                                    await self.exchange.cancel_order(trade.sell_order_id, trade.symbol)
                                    sell_order = await self.exchange.create_order(
                                        trade.symbol,
                                        OrderType.LIMIT,
                                        OrderSide.SELL,
                                        trade.size,
                                        price=new_tp
                                    )
                                    if sell_order:
                                        trade.sell_order_id = sell_order.id
                                        trade.take_profit = new_tp
                                        logger.info(f"{trade.symbol} sell order raised: ${new_tp:.6f}")
                            except Exception as e:
                                logger.debug(f"Sell order adjustment error: {e}")
                    
                    profit_from_entry = (current_price - trade.entry_price) / trade.entry_price
                    drop_from_peak = (trade.peak_price - current_price) / trade.peak_price if trade.peak_price > 0 else 0
                    
                    if trade.breakout_confirmed:
                        trailing_drop_pct = 0.08
                        min_profit_pct = 0.15
                    else:
                        trailing_drop_pct = 0.10
                        min_profit_pct = 0.20
                    
                    if profit_from_entry >= min_profit_pct and drop_from_peak >= trailing_drop_pct:
                        should_close = True
                        close_reason = f"Trailing Peak Sell (peak ${trade.peak_price:.6f}, dropped {drop_from_peak:.1%})"
                    
                    if trade.fakeout_signals >= 10 and profit_from_entry > 0.10:
                        should_close = True
                        close_reason = f"Fakeout Protection ({trade.fakeout_signals} weak peaks detected)"
                    
                    if len(trade.price_history) >= 20:
                        recent_prices = trade.price_history[-20:]
                        declining_count = sum(1 for i in range(1, len(recent_prices)) if recent_prices[i] < recent_prices[i-1])
                        if declining_count >= 18 and profit_from_entry > 0.08:
                            should_close = True
                            close_reason = "Momentum Reversal (18+ declining ticks)"
                    
                    if trade.breakout_confirmed and drop_from_peak >= 0.15 and profit_from_entry >= 0.10:
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
                    
                    if profit_from_entry >= 0.15 and rise_from_peak >= 0.08:
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
            exit_time = datetime.now(LA_TZ)
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
            entry_hour = entry_time.hour
            self.telegram.learning_engine.record_trade_outcome(
                symbol=closed_trade.symbol,
                pnl=closed_trade.pnl,
                pnl_pct=pnl_pct,
                hold_duration_mins=duration // 60,
                strategy=closed_trade.strategy,
                entry_hour=entry_hour
            )
            trade_metrics = TradeMetrics(
                symbol=closed_trade.symbol,
                pnl=closed_trade.pnl,
                pnl_pct=pnl_pct,
                hold_time_seconds=duration,
                entry_price=closed_trade.entry_price,
                exit_price=exit_price,
                size=closed_trade.size,
                confidence=closed_trade.confidence,
                timestamp=exit_time.isoformat()
            )
            self.kpi_tracker.record_trade(trade_metrics)
            self.kpi_tracker.update_equity(self.state.state.equity)
            logger.info(f"Trade closed: {closed_trade.symbol} PnL: ${closed_trade.pnl:.2f} ({pnl_pct:+.2f}%)")

    async def run_priority_cycle(self) -> None:
        try:
            await self._refresh_balance()
            candidates = await self.scan_priority_candidates()
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
                        tg_should_alert = self.telegram.learning_engine.should_alert_for_symbol(
                            signal.symbol, signal.confidence
                        )
                        is_preferred_hour = self.telegram.learning_engine.is_preferred_trading_hour()
                        if not tg_should_alert:
                            ml_reasons.append("Telegram learning: symbol underperforming")
                            adjusted_conf *= 0.85
                        if not is_preferred_hour:
                            ml_reasons.append("Outside preferred trading hours")
                            adjusted_conf *= 0.95
                        if should_trade and adjusted_conf >= self.state.state.config.min_confidence:
                            signal.confidence = adjusted_conf
                            if ml_reasons:
                                signal.reasons.extend(ml_reasons)
                            await self.execute_trade(signal)
                        else:
                            block_reason = "should_trade=False" if not should_trade else f"confidence {adjusted_conf:.1f} < min {self.state.state.config.min_confidence}"
                            logger.info(f"[TRADE BLOCKED] ML blocked {signal.symbol}: {block_reason} | Details: {ml_reasons}")
            await self.manage_positions()
        except Exception as e:
            logger.error(f"Priority cycle error: {e}")

    async def run_cycle(self) -> None:
        try:
            await self._refresh_balance()
            await self.ensure_sell_orders_placed()
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
                        tg_should_alert = self.telegram.learning_engine.should_alert_for_symbol(
                            signal.symbol, signal.confidence
                        )
                        is_preferred_hour = self.telegram.learning_engine.is_preferred_trading_hour()
                        if not tg_should_alert:
                            ml_reasons.append("Telegram learning: symbol underperforming")
                            adjusted_conf *= 0.85
                        if not is_preferred_hour:
                            ml_reasons.append("Outside preferred trading hours")
                            adjusted_conf *= 0.95
                        if should_trade and adjusted_conf >= self.state.state.config.min_confidence:
                            signal.confidence = adjusted_conf
                            if ml_reasons:
                                signal.reasons.extend(ml_reasons)
                            await self.execute_trade(signal)
                        else:
                            block_reason = "should_trade=False" if not should_trade else f"confidence {adjusted_conf:.1f} < min {self.state.state.config.min_confidence}"
                            logger.info(f"[TRADE BLOCKED] ML blocked {signal.symbol}: {block_reason} | Details: {ml_reasons}")
                    else:
                        logger.info(f"Max positions ({self.state.state.config.max_open_positions}) reached, monitoring existing trades")
            await self.manage_positions()
            self._learning_cycle_counter += 1
            if self._learning_cycle_counter >= 100:
                self._learning_cycle_counter = 0
                result = self.ml_engine.run_learning_cycle()
                logger.info(f"ML Learning cycle: {result}")
                tg_result = self.telegram.learning_engine.run_learning_cycle()
                logger.info(f"Telegram Learning cycle: {tg_result}")
            await self.telegram.check_scheduled_reports()
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
                now = time.time()
                full_scan_interval = self.state.state.config.scan_interval
                position_interval = self.state.state.config.position_scan_interval
                should_full_scan = (now - self._last_full_scan) >= full_scan_interval
                should_position_scan = (now - self._last_position_scan) >= position_interval
                priority_scan_interval = 120
                should_priority_scan = self._priority_candidates and (now - self._last_priority_scan) >= priority_scan_interval
                if should_full_scan:
                    logger.info("Running full market scan cycle...")
                    await self.run_cycle()
                    self._last_priority_scan = now
                elif should_priority_scan:
                    logger.info(f"Running priority candidates cycle (top {len(self._priority_candidates)})...")
                    await self.run_priority_cycle()
                    self._last_priority_scan = now
                elif should_position_scan:
                    await self._refresh_balance()
                    await self.scan_active_positions()
                    await self.manage_positions()
                    self._last_position_scan = now
                    logger.info("Position scan cycle complete.")
                else:
                    await self.manage_positions()
                next_position = max(0, position_interval - (time.time() - self._last_position_scan))
                next_full = max(0, full_scan_interval - (time.time() - self._last_full_scan))
                sleep_time = max(5, min(next_position, next_full, 30))
                logger.info(f"Next position scan: {int(next_position)}s | Next full scan: {int(next_full)}s | Sleeping {int(sleep_time)}s...")
                await asyncio.sleep(sleep_time)
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
