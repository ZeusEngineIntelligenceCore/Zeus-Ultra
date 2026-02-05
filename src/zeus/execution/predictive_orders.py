#!/usr/bin/env python3
"""
ZEUS PREDICTIVE LIMIT ORDER ENGINE
Predicts optimal entry prices by analyzing support levels, order book depth,
momentum exhaustion, and price reversal patterns to find the lowest possible
entry price before the coin reverses upward.
"""

import logging
import statistics
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone
import numpy as np

logger = logging.getLogger("Zeus.PredictiveOrders")


@dataclass
class SupportLevel:
    price: float
    strength: float
    touches: int
    recency_score: float
    volume_at_level: float
    
    @property
    def combined_score(self) -> float:
        return (self.strength * 0.3 + 
                min(self.touches / 5, 1.0) * 0.25 + 
                self.recency_score * 0.25 + 
                min(self.volume_at_level / 1000, 1.0) * 0.2)


@dataclass
class MomentumExhaustion:
    is_exhausted: bool
    exhaustion_score: float
    reversal_probability: float
    expected_bounce_pct: float
    time_to_reversal_mins: int
    confidence: float


@dataclass
class OrderBookAnalysis:
    best_bid: float
    best_ask: float
    spread_pct: float
    bid_wall_price: Optional[float]
    bid_wall_size: float
    ask_pressure: float
    bid_support_depth: float
    imbalance_ratio: float
    smart_money_accumulation: bool
    optimal_limit_price: float


@dataclass
class PredictedEntry:
    symbol: str
    current_price: float
    predicted_low: float
    optimal_limit_price: float
    confidence: float
    savings_pct: float
    support_levels: List[SupportLevel]
    momentum_analysis: MomentumExhaustion
    order_book: OrderBookAnalysis
    time_window_mins: int
    recommendation: str
    risk_score: float


class PredictiveLimitOrderEngine:
    def __init__(self):
        self.price_history_cache: Dict[str, List[float]] = {}
        self.volume_history_cache: Dict[str, List[float]] = {}
        self.prediction_accuracy: Dict[str, List[bool]] = {}
        self.support_cache: Dict[str, List[SupportLevel]] = {}
        
    def identify_support_levels(
        self,
        prices: List[float],
        volumes: List[float],
        current_price: float,
        lookback: int = 100
    ) -> List[SupportLevel]:
        if len(prices) < lookback:
            return []
        
        recent_prices = prices[-lookback:]
        recent_volumes = volumes[-lookback:] if len(volumes) >= lookback else [1.0] * lookback
        
        price_range = max(recent_prices) - min(recent_prices)
        if price_range == 0:
            return []
        
        num_bins = 20
        bin_size = price_range / num_bins
        min_price = min(recent_prices)
        
        bins: Dict[int, Dict[str, Any]] = {}
        
        for i, (price, volume) in enumerate(zip(recent_prices, recent_volumes)):
            bin_idx = int((price - min_price) / bin_size)
            bin_idx = min(bin_idx, num_bins - 1)
            
            if bin_idx not in bins:
                bins[bin_idx] = {"prices": [], "volumes": [], "indices": []}
            bins[bin_idx]["prices"].append(price)
            bins[bin_idx]["volumes"].append(volume)
            bins[bin_idx]["indices"].append(i)
        
        support_levels = []
        for bin_idx, data in bins.items():
            if len(data["prices"]) < 3:
                continue
            
            bin_price = min_price + (bin_idx + 0.5) * bin_size
            
            if bin_price >= current_price:
                continue
            
            touches = len(data["prices"])
            avg_volume = statistics.mean(data["volumes"]) if data["volumes"] else 1.0
            recency = max(data["indices"]) / lookback
            
            price_reversals = 0
            for idx in data["indices"]:
                if idx < lookback - 1:
                    if recent_prices[idx + 1] > recent_prices[idx]:
                        price_reversals += 1
            
            strength = min(price_reversals / max(touches, 1), 1.0)
            
            support_levels.append(SupportLevel(
                price=bin_price,
                strength=strength,
                touches=touches,
                recency_score=recency,
                volume_at_level=avg_volume
            ))
        
        support_levels.sort(key=lambda x: x.combined_score, reverse=True)
        return support_levels[:5]
    
    def detect_momentum_exhaustion(
        self,
        prices: List[float],
        volumes: List[float],
        rsi: float,
        macd: float,
        macd_signal: float,
        atr: float
    ) -> MomentumExhaustion:
        if len(prices) < 20:
            return MomentumExhaustion(
                is_exhausted=False,
                exhaustion_score=0.0,
                reversal_probability=0.0,
                expected_bounce_pct=0.0,
                time_to_reversal_mins=0,
                confidence=0.0
            )
        
        exhaustion_signals = 0
        total_weight = 0
        
        if rsi < 30:
            exhaustion_signals += (30 - rsi) / 30 * 3
            total_weight += 3
        elif rsi < 40:
            exhaustion_signals += (40 - rsi) / 40 * 1.5
            total_weight += 1.5
        else:
            total_weight += 3
        
        if macd > macd_signal and macd < 0:
            exhaustion_signals += 2
        total_weight += 2
        
        recent_prices = prices[-10:]
        price_changes = [recent_prices[i] - recent_prices[i-1] for i in range(1, len(recent_prices))]
        negative_momentum = sum(1 for c in price_changes if c < 0)
        
        if negative_momentum >= 7:
            trend_exhaustion = 1.0
        elif negative_momentum >= 5:
            trend_exhaustion = 0.7
        else:
            trend_exhaustion = 0.3
        
        exhaustion_signals += trend_exhaustion * 2
        total_weight += 2
        
        if len(volumes) >= 10:
            recent_vol = volumes[-5:]
            older_vol = volumes[-10:-5]
            if statistics.mean(recent_vol) < statistics.mean(older_vol) * 0.7:
                exhaustion_signals += 1.5
            total_weight += 1.5
        
        current_price = prices[-1]
        recent_low = min(prices[-20:])
        distance_from_low = (current_price - recent_low) / recent_low if recent_low > 0 else 0
        
        if distance_from_low < 0.02:
            exhaustion_signals += 1.5
            total_weight += 1.5
        
        exhaustion_score = exhaustion_signals / total_weight if total_weight > 0 else 0
        is_exhausted = exhaustion_score > 0.6
        
        reversal_probability = min(exhaustion_score * 1.2, 0.95)
        expected_bounce_pct = atr / current_price * 2 if current_price > 0 else 0.02
        
        if exhaustion_score > 0.8:
            time_to_reversal = 5
        elif exhaustion_score > 0.6:
            time_to_reversal = 15
        elif exhaustion_score > 0.4:
            time_to_reversal = 30
        else:
            time_to_reversal = 60
        
        confidence = min(exhaustion_score * 0.9 + 0.1, 0.95)
        
        return MomentumExhaustion(
            is_exhausted=is_exhausted,
            exhaustion_score=exhaustion_score,
            reversal_probability=reversal_probability,
            expected_bounce_pct=expected_bounce_pct,
            time_to_reversal_mins=time_to_reversal,
            confidence=confidence
        )
    
    def analyze_order_book(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        current_price: float
    ) -> OrderBookAnalysis:
        if not bids or not asks:
            return OrderBookAnalysis(
                best_bid=current_price * 0.999,
                best_ask=current_price * 1.001,
                spread_pct=0.2,
                bid_wall_price=None,
                bid_wall_size=0,
                ask_pressure=0.5,
                bid_support_depth=0,
                imbalance_ratio=1.0,
                smart_money_accumulation=False,
                optimal_limit_price=current_price * 0.998
            )
        
        best_bid = bids[0][0] if bids else current_price * 0.999
        best_ask = asks[0][0] if asks else current_price * 1.001
        spread_pct = ((best_ask - best_bid) / best_bid) * 100 if best_bid > 0 else 0.2
        
        bid_wall_price = None
        bid_wall_size = 0
        avg_bid_size = statistics.mean([b[1] for b in bids[:10]]) if len(bids) >= 10 else 1
        
        for price, size in bids[:20]:
            if size > avg_bid_size * 3:
                bid_wall_price = price
                bid_wall_size = size
                break
        
        total_bid_volume = sum(b[1] for b in bids[:15])
        total_ask_volume = sum(a[1] for a in asks[:15])
        
        if total_bid_volume + total_ask_volume > 0:
            ask_pressure = total_ask_volume / (total_bid_volume + total_ask_volume)
        else:
            ask_pressure = 0.5
        
        bid_support_depth = total_bid_volume
        imbalance_ratio = total_bid_volume / total_ask_volume if total_ask_volume > 0 else 1.0
        
        smart_money_accumulation = (
            imbalance_ratio > 1.5 and 
            bid_wall_price is not None and 
            bid_wall_size > avg_bid_size * 5
        )
        
        if bid_wall_price and bid_wall_price < current_price * 0.995:
            optimal_limit_price = bid_wall_price * 1.001
        elif spread_pct > 0.3:
            optimal_limit_price = best_bid + (best_ask - best_bid) * 0.3
        else:
            optimal_limit_price = best_bid * 1.0005
        
        return OrderBookAnalysis(
            best_bid=best_bid,
            best_ask=best_ask,
            spread_pct=spread_pct,
            bid_wall_price=bid_wall_price,
            bid_wall_size=bid_wall_size,
            ask_pressure=ask_pressure,
            bid_support_depth=bid_support_depth,
            imbalance_ratio=imbalance_ratio,
            smart_money_accumulation=smart_money_accumulation,
            optimal_limit_price=optimal_limit_price
        )
    
    def predict_optimal_entry(
        self,
        symbol: str,
        current_price: float,
        prices: List[float],
        volumes: List[float],
        rsi: float,
        macd: float,
        macd_signal: float,
        atr: float,
        bids: Optional[List[Tuple[float, float]]] = None,
        asks: Optional[List[Tuple[float, float]]] = None
    ) -> PredictedEntry:
        support_levels = self.identify_support_levels(prices, volumes, current_price)
        self.support_cache[symbol] = support_levels
        
        momentum = self.detect_momentum_exhaustion(
            prices, volumes, rsi, macd, macd_signal, atr
        )
        
        order_book = self.analyze_order_book(
            bids or [], asks or [], current_price
        )
        
        predicted_low = current_price
        confidence_factors = []
        
        if support_levels:
            nearest_support = max(
                [s for s in support_levels if s.price < current_price * 0.99],
                key=lambda x: x.combined_score,
                default=None
            )
            if nearest_support:
                support_low = nearest_support.price
                support_confidence = nearest_support.combined_score
                confidence_factors.append(("support", support_confidence, support_low))
        
        if momentum.is_exhausted:
            momentum_low = current_price * (1 - momentum.expected_bounce_pct * 0.3)
            confidence_factors.append(("momentum", momentum.confidence, momentum_low))
        
        if order_book.bid_wall_price:
            orderbook_low = order_book.bid_wall_price
            ob_confidence = min(order_book.bid_wall_size / 10000, 0.9)
            confidence_factors.append(("orderbook", ob_confidence, orderbook_low))
        
        if not confidence_factors:
            predicted_low = current_price * 0.998
            optimal_limit = predicted_low
            confidence = 0.3
        else:
            total_weight = sum(cf[1] for cf in confidence_factors)
            predicted_low = sum(cf[1] * cf[2] for cf in confidence_factors) / total_weight
            confidence = min(statistics.mean([cf[1] for cf in confidence_factors]) + 0.1, 0.95)
            
            optimal_limit = max(predicted_low, order_book.optimal_limit_price)
            optimal_limit = max(optimal_limit, current_price * 0.97)
        
        savings_pct = ((current_price - optimal_limit) / current_price) * 100 if current_price > 0 else 0
        
        if confidence > 0.8 and savings_pct > 1.0:
            recommendation = "STRONG_LIMIT"
        elif confidence > 0.6 and savings_pct > 0.5:
            recommendation = "LIMIT"
        elif momentum.is_exhausted:
            recommendation = "WAIT_FOR_REVERSAL"
        else:
            recommendation = "MARKET_IF_URGENT"
        
        risk_score = 1.0 - confidence
        if order_book.ask_pressure > 0.7:
            risk_score += 0.15
        if not momentum.is_exhausted and rsi > 50:
            risk_score += 0.1
        risk_score = min(risk_score, 1.0)
        
        time_window = momentum.time_to_reversal_mins if momentum.is_exhausted else 30
        
        return PredictedEntry(
            symbol=symbol,
            current_price=current_price,
            predicted_low=predicted_low,
            optimal_limit_price=optimal_limit,
            confidence=confidence,
            savings_pct=savings_pct,
            support_levels=support_levels,
            momentum_analysis=momentum,
            order_book=order_book,
            time_window_mins=time_window,
            recommendation=recommendation,
            risk_score=risk_score
        )
    
    def get_limit_order_params(
        self,
        prediction: PredictedEntry,
        urgency: str = "normal"
    ) -> Dict[str, Any]:
        if urgency == "high" or prediction.recommendation == "MARKET_IF_URGENT":
            price = prediction.current_price * 0.999
            order_type = "limit"
        elif prediction.recommendation == "STRONG_LIMIT":
            price = prediction.optimal_limit_price
            order_type = "limit"
        elif prediction.recommendation == "LIMIT":
            price = (prediction.optimal_limit_price + prediction.current_price * 0.999) / 2
            order_type = "limit"
        else:
            price = prediction.optimal_limit_price
            order_type = "limit"
        
        return {
            "order_type": order_type,
            "price": price,
            "time_in_force": "GTC",
            "predicted_savings_pct": prediction.savings_pct,
            "confidence": prediction.confidence,
            "time_window_mins": prediction.time_window_mins,
            "recommendation": prediction.recommendation
        }
    
    def record_prediction_result(
        self,
        symbol: str,
        predicted_low: float,
        actual_low: float,
        timeframe_mins: int
    ) -> None:
        accuracy_threshold = 0.02
        was_accurate = abs(predicted_low - actual_low) / actual_low < accuracy_threshold
        
        if symbol not in self.prediction_accuracy:
            self.prediction_accuracy[symbol] = []
        
        self.prediction_accuracy[symbol].append(was_accurate)
        self.prediction_accuracy[symbol] = self.prediction_accuracy[symbol][-100:]
        
        if was_accurate:
            logger.info(f"Prediction accurate for {symbol}: predicted {predicted_low:.6f}, actual {actual_low:.6f}")
        else:
            logger.debug(f"Prediction miss for {symbol}: predicted {predicted_low:.6f}, actual {actual_low:.6f}")
    
    def get_prediction_accuracy(self, symbol: str) -> float:
        if symbol not in self.prediction_accuracy:
            return 0.5
        
        history = self.prediction_accuracy[symbol]
        if not history:
            return 0.5
        
        return sum(history) / len(history)
