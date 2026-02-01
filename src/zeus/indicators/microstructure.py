#!/usr/bin/env python3
"""
ZEUS MARKET MICROSTRUCTURE ANALYZER
Advanced order flow analysis for institutional-grade trading signals

Features:
- Order Flow Imbalance (OFI) - Detects buying/selling pressure
- VPIN (Volume-Synchronized Probability of Informed Trading) - Toxicity detection
- Spread Analysis - Liquidity assessment
- Iceberg Detection - Hidden order identification
- Market Maker Positioning - Smart money signals
"""

from __future__ import annotations
import math
from typing import List, Dict, Any, Optional, Tuple, Deque
from dataclasses import dataclass, field
from collections import deque
import numpy as np


@dataclass
class OrderBookSnapshot:
    timestamp: float
    bids: List[Tuple[float, float]]
    asks: List[Tuple[float, float]]
    mid_price: float = 0.0
    spread: float = 0.0
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    
    def __post_init__(self):
        if self.bids and self.asks:
            best_bid = self.bids[0][0] if self.bids else 0
            best_ask = self.asks[0][0] if self.asks else 0
            self.mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
            self.spread = best_ask - best_bid if best_bid and best_ask else 0
            self.bid_depth = sum(qty for _, qty in self.bids[:10])
            self.ask_depth = sum(qty for _, qty in self.asks[:10])


@dataclass
class TradeFlow:
    price: float
    size: float
    side: str
    timestamp: float


@dataclass
class MicrostructureConfig:
    ofi_window: int = 50
    vpin_buckets: int = 50
    vpin_bucket_size: float = 0.0
    spread_ma_period: int = 20
    depth_imbalance_threshold: float = 0.3
    iceberg_detection_window: int = 20
    iceberg_repeat_threshold: int = 3
    toxicity_threshold: float = 0.7
    liquidity_score_depth: int = 10


@dataclass 
class MicrostructureSignal:
    ofi_score: float = 0.0
    vpin_toxicity: float = 0.0
    spread_percentile: float = 0.0
    depth_imbalance: float = 0.0
    iceberg_probability: float = 0.0
    liquidity_score: float = 0.0
    market_maker_bias: float = 0.0
    trade_flow_intensity: float = 0.0
    smart_money_signal: float = 0.0
    overall_score: float = 0.0
    signals: List[str] = field(default_factory=list)


class MicrostructureAnalyzer:
    def __init__(self, config: Optional[MicrostructureConfig] = None):
        self.config = config or MicrostructureConfig()
        self.order_book_history: Deque[OrderBookSnapshot] = deque(maxlen=100)
        self.trade_history: Deque[TradeFlow] = deque(maxlen=500)
        self.spread_history: Deque[float] = deque(maxlen=100)
        self.vpin_buckets: List[Tuple[float, float]] = []
        self.volume_bucket_target = 0.0
        self.current_bucket_volume = 0.0
        self.current_bucket_buy_volume = 0.0
        
    def update_order_book(self, snapshot: OrderBookSnapshot) -> None:
        self.order_book_history.append(snapshot)
        if snapshot.spread > 0:
            self.spread_history.append(snapshot.spread)
    
    def update_trade(self, trade: TradeFlow) -> None:
        self.trade_history.append(trade)
        self._update_vpin_bucket(trade)
    
    def _update_vpin_bucket(self, trade: TradeFlow) -> None:
        self.current_bucket_volume += trade.size
        if trade.side == "buy":
            self.current_bucket_buy_volume += trade.size
        
        if self.volume_bucket_target > 0 and self.current_bucket_volume >= self.volume_bucket_target:
            buy_ratio = self.current_bucket_buy_volume / self.current_bucket_volume if self.current_bucket_volume > 0 else 0.5
            self.vpin_buckets.append((buy_ratio, self.current_bucket_volume))
            if len(self.vpin_buckets) > self.config.vpin_buckets:
                self.vpin_buckets.pop(0)
            self.current_bucket_volume = 0.0
            self.current_bucket_buy_volume = 0.0
    
    def calculate_ofi(self) -> float:
        if len(self.order_book_history) < 2:
            return 0.0
        
        ofi_sum = 0.0
        snapshots = list(self.order_book_history)[-self.config.ofi_window:]
        
        for i in range(1, len(snapshots)):
            prev = snapshots[i - 1]
            curr = snapshots[i]
            
            if not prev.bids or not prev.asks or not curr.bids or not curr.asks:
                continue
            
            prev_best_bid, prev_bid_qty = prev.bids[0]
            curr_best_bid, curr_bid_qty = curr.bids[0]
            prev_best_ask, prev_ask_qty = prev.asks[0]
            curr_best_ask, curr_ask_qty = curr.asks[0]
            
            if curr_best_bid >= prev_best_bid:
                bid_ofi = curr_bid_qty if curr_best_bid > prev_best_bid else curr_bid_qty - prev_bid_qty
            else:
                bid_ofi = -prev_bid_qty
            
            if curr_best_ask <= prev_best_ask:
                ask_ofi = -curr_ask_qty if curr_best_ask < prev_best_ask else -(curr_ask_qty - prev_ask_qty)
            else:
                ask_ofi = prev_ask_qty
            
            ofi_sum += bid_ofi + ask_ofi
        
        max_ofi = sum(s.bid_depth + s.ask_depth for s in snapshots) / len(snapshots) if snapshots else 1.0
        normalized_ofi = ofi_sum / max_ofi if max_ofi > 0 else 0.0
        
        return max(-1.0, min(1.0, normalized_ofi))
    
    def calculate_vpin(self) -> float:
        if len(self.vpin_buckets) < 10:
            return 0.5
        
        abs_imbalances = []
        for buy_ratio, volume in self.vpin_buckets:
            imbalance = abs(buy_ratio - 0.5) * 2
            abs_imbalances.append(imbalance)
        
        vpin = sum(abs_imbalances) / len(abs_imbalances) if abs_imbalances else 0.5
        return min(1.0, vpin)
    
    def calculate_spread_analysis(self) -> Tuple[float, float]:
        if not self.spread_history:
            return 0.5, 0.0
        
        spreads = list(self.spread_history)
        current_spread = spreads[-1]
        
        sorted_spreads = sorted(spreads)
        rank = sorted_spreads.index(current_spread) if current_spread in sorted_spreads else len(sorted_spreads) // 2
        percentile = rank / len(sorted_spreads) if sorted_spreads else 0.5
        
        avg_spread = sum(spreads) / len(spreads)
        spread_volatility = math.sqrt(sum((s - avg_spread) ** 2 for s in spreads) / len(spreads)) if len(spreads) > 1 else 0
        
        return percentile, spread_volatility
    
    def calculate_depth_imbalance(self) -> float:
        if not self.order_book_history:
            return 0.0
        
        latest = self.order_book_history[-1]
        total_depth = latest.bid_depth + latest.ask_depth
        
        if total_depth == 0:
            return 0.0
        
        imbalance = (latest.bid_depth - latest.ask_depth) / total_depth
        return max(-1.0, min(1.0, imbalance))
    
    def detect_iceberg_orders(self) -> float:
        if len(self.order_book_history) < self.config.iceberg_detection_window:
            return 0.0
        
        snapshots = list(self.order_book_history)[-self.config.iceberg_detection_window:]
        
        bid_repeats = {}
        ask_repeats = {}
        
        for snapshot in snapshots:
            for price, qty in snapshot.bids[:5]:
                key = (round(price, 8), round(qty, 4))
                bid_repeats[key] = bid_repeats.get(key, 0) + 1
            
            for price, qty in snapshot.asks[:5]:
                key = (round(price, 8), round(qty, 4))
                ask_repeats[key] = ask_repeats.get(key, 0) + 1
        
        iceberg_signals = 0
        for count in list(bid_repeats.values()) + list(ask_repeats.values()):
            if count >= self.config.iceberg_repeat_threshold:
                iceberg_signals += 1
        
        max_signals = len(snapshots) * 10
        probability = min(1.0, iceberg_signals / max_signals * 5) if max_signals > 0 else 0.0
        
        return probability
    
    def calculate_liquidity_score(self) -> float:
        if not self.order_book_history:
            return 0.5
        
        latest = self.order_book_history[-1]
        
        if not latest.bids or not latest.asks:
            return 0.5
        
        bid_liquidity = sum(price * qty for price, qty in latest.bids[:self.config.liquidity_score_depth])
        ask_liquidity = sum(price * qty for price, qty in latest.asks[:self.config.liquidity_score_depth])
        
        total_liquidity = bid_liquidity + ask_liquidity
        
        spread_factor = 1.0 - min(1.0, (latest.spread / latest.mid_price * 100) / 2) if latest.mid_price > 0 else 0.5
        
        depth_factor = min(1.0, total_liquidity / 100000)
        
        liquidity_score = (spread_factor * 0.6 + depth_factor * 0.4)
        
        return max(0.0, min(1.0, liquidity_score))
    
    def calculate_market_maker_bias(self) -> float:
        if len(self.order_book_history) < 10:
            return 0.0
        
        snapshots = list(self.order_book_history)[-20:]
        
        bid_pressure_sum = 0.0
        ask_pressure_sum = 0.0
        
        for snapshot in snapshots:
            if snapshot.bids and snapshot.asks:
                weighted_bid = sum((price * qty) for price, qty in snapshot.bids[:5])
                weighted_ask = sum((price * qty) for price, qty in snapshot.asks[:5])
                
                total = weighted_bid + weighted_ask
                if total > 0:
                    bid_pressure_sum += weighted_bid / total
                    ask_pressure_sum += weighted_ask / total
        
        if len(snapshots) == 0:
            return 0.0
        
        avg_bid_pressure = bid_pressure_sum / len(snapshots)
        avg_ask_pressure = ask_pressure_sum / len(snapshots)
        
        bias = (avg_bid_pressure - avg_ask_pressure) * 2
        return max(-1.0, min(1.0, bias))
    
    def calculate_trade_flow_intensity(self) -> float:
        if len(self.trade_history) < 10:
            return 0.5
        
        recent_trades = list(self.trade_history)[-50:]
        
        if len(recent_trades) < 2:
            return 0.5
        
        time_span = recent_trades[-1].timestamp - recent_trades[0].timestamp
        if time_span <= 0:
            return 0.5
        
        trade_rate = len(recent_trades) / time_span
        total_volume = sum(t.size for t in recent_trades)
        volume_rate = total_volume / time_span
        
        intensity = min(1.0, trade_rate / 10) * 0.5 + min(1.0, volume_rate / 1000) * 0.5
        
        return intensity
    
    def calculate_smart_money_signal(self) -> float:
        if len(self.trade_history) < 20:
            return 0.0
        
        recent_trades = list(self.trade_history)[-100:]
        
        volumes = [t.size for t in recent_trades]
        avg_volume = sum(volumes) / len(volumes)
        std_volume = math.sqrt(sum((v - avg_volume) ** 2 for v in volumes) / len(volumes)) if len(volumes) > 1 else avg_volume
        
        large_trade_threshold = avg_volume + 2 * std_volume
        
        large_buys = sum(t.size for t in recent_trades if t.side == "buy" and t.size > large_trade_threshold)
        large_sells = sum(t.size for t in recent_trades if t.side == "sell" and t.size > large_trade_threshold)
        
        total_large = large_buys + large_sells
        if total_large == 0:
            return 0.0
        
        smart_money = (large_buys - large_sells) / total_large
        return max(-1.0, min(1.0, smart_money))
    
    def analyze(self, order_book: Optional[OrderBookSnapshot] = None, 
                trades: Optional[List[TradeFlow]] = None) -> MicrostructureSignal:
        if order_book:
            self.update_order_book(order_book)
        
        if trades:
            for trade in trades:
                self.update_trade(trade)
        
        ofi = self.calculate_ofi()
        vpin = self.calculate_vpin()
        spread_percentile, spread_vol = self.calculate_spread_analysis()
        depth_imbalance = self.calculate_depth_imbalance()
        iceberg_prob = self.detect_iceberg_orders()
        liquidity = self.calculate_liquidity_score()
        mm_bias = self.calculate_market_maker_bias()
        flow_intensity = self.calculate_trade_flow_intensity()
        smart_money = self.calculate_smart_money_signal()
        
        signals = []
        
        overall = 50.0
        
        if ofi > 0.3:
            overall += ofi * 15
            signals.append(f"Strong buying pressure (OFI: {ofi:.2f})")
        elif ofi < -0.3:
            overall += ofi * 15
            signals.append(f"Strong selling pressure (OFI: {ofi:.2f})")
        
        if vpin > self.config.toxicity_threshold:
            overall -= (vpin - 0.5) * 20
            signals.append(f"High toxicity detected (VPIN: {vpin:.2f})")
        
        if depth_imbalance > 0.3:
            overall += depth_imbalance * 10
            signals.append(f"Bid-heavy order book ({depth_imbalance:.1%})")
        elif depth_imbalance < -0.3:
            overall += depth_imbalance * 10
            signals.append(f"Ask-heavy order book ({depth_imbalance:.1%})")
        
        if iceberg_prob > 0.5:
            signals.append(f"Iceberg orders detected ({iceberg_prob:.1%})")
            overall += 5
        
        if liquidity < 0.3:
            overall -= (0.5 - liquidity) * 15
            signals.append("Low liquidity - caution advised")
        elif liquidity > 0.7:
            overall += (liquidity - 0.5) * 10
            signals.append("High liquidity - favorable execution")
        
        if mm_bias > 0.3:
            overall += mm_bias * 10
            signals.append(f"Market makers bullish ({mm_bias:.1%})")
        elif mm_bias < -0.3:
            overall += mm_bias * 10
            signals.append(f"Market makers bearish ({mm_bias:.1%})")
        
        if smart_money > 0.4:
            overall += smart_money * 15
            signals.append(f"Smart money buying ({smart_money:.1%})")
        elif smart_money < -0.4:
            overall += smart_money * 15
            signals.append(f"Smart money selling ({smart_money:.1%})")
        
        overall = max(0.0, min(100.0, overall))
        
        return MicrostructureSignal(
            ofi_score=ofi,
            vpin_toxicity=vpin,
            spread_percentile=spread_percentile,
            depth_imbalance=depth_imbalance,
            iceberg_probability=iceberg_prob,
            liquidity_score=liquidity,
            market_maker_bias=mm_bias,
            trade_flow_intensity=flow_intensity,
            smart_money_signal=smart_money,
            overall_score=overall,
            signals=signals
        )
    
    def set_volume_bucket_target(self, daily_volume: float) -> None:
        self.volume_bucket_target = daily_volume / self.config.vpin_buckets if daily_volume > 0 else 0


class OrderFlowTracker:
    def __init__(self):
        self.symbol_analyzers: Dict[str, MicrostructureAnalyzer] = {}
    
    def get_analyzer(self, symbol: str) -> MicrostructureAnalyzer:
        if symbol not in self.symbol_analyzers:
            self.symbol_analyzers[symbol] = MicrostructureAnalyzer()
        return self.symbol_analyzers[symbol]
    
    def analyze_symbol(self, symbol: str, order_book: Optional[OrderBookSnapshot] = None,
                      trades: Optional[List[TradeFlow]] = None) -> MicrostructureSignal:
        analyzer = self.get_analyzer(symbol)
        return analyzer.analyze(order_book, trades)
    
    def cleanup_old_symbols(self, active_symbols: List[str]) -> None:
        to_remove = [s for s in self.symbol_analyzers if s not in active_symbols]
        for symbol in to_remove:
            del self.symbol_analyzers[symbol]
