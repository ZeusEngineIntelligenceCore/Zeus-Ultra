#!/usr/bin/env python3
"""
ZEUS SMART ORDER EXECUTION ENGINE
Institutional-grade execution algorithms:
- TWAP (Time-Weighted Average Price)
- VWAP (Volume-Weighted Average Price)
- Iceberg Orders
- Slippage Prediction & Minimization
- Optimal Timing Based on Liquidity
"""

from __future__ import annotations
import asyncio
import math
import time
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta


class ExecutionStrategy(Enum):
    MARKET = "market"
    LIMIT = "limit"
    TWAP = "twap"
    VWAP = "vwap"
    ICEBERG = "iceberg"
    ADAPTIVE = "adaptive"


@dataclass
class ExecutionConfig:
    twap_slices: int = 10
    twap_interval_seconds: int = 30
    vwap_participation_rate: float = 0.10
    iceberg_show_pct: float = 0.20
    max_slippage_pct: float = 0.005
    urgency_factor: float = 0.5
    min_slice_value: float = 10.0
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    price_improvement_wait_ms: int = 500


@dataclass
class OrderSlice:
    sequence: int
    size: float
    price: Optional[float]
    order_type: str
    status: str = "pending"
    order_id: Optional[str] = None
    filled_size: float = 0.0
    filled_price: float = 0.0
    created_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None


@dataclass
class ExecutionPlan:
    strategy: ExecutionStrategy
    total_size: float
    side: str
    symbol: str
    slices: List[OrderSlice] = field(default_factory=list)
    target_price: float = 0.0
    price_limit: float = 0.0
    estimated_slippage: float = 0.0
    estimated_cost: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


@dataclass
class ExecutionResult:
    success: bool
    total_filled: float
    average_price: float
    total_cost: float
    slippage: float
    slippage_cost: float
    execution_time_seconds: float
    slices_completed: int
    slices_failed: int
    details: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MarketConditions:
    spread: float
    spread_pct: float
    bid_depth: float
    ask_depth: float
    volatility: float
    volume_24h: float
    recent_volume: float
    liquidity_score: float


class SlippagePredictor:
    def __init__(self):
        self.historical_slippage: Dict[str, List[Tuple[float, float, float]]] = {}
    
    def predict_slippage(self, symbol: str, order_size: float, 
                        market_conditions: MarketConditions) -> float:
        base_slippage = market_conditions.spread_pct / 2
        
        size_impact = 0.0
        if market_conditions.bid_depth > 0:
            size_ratio = order_size / market_conditions.bid_depth
            size_impact = size_ratio * 0.5
        
        volatility_impact = market_conditions.volatility * 0.1
        
        liquidity_impact = (1 - market_conditions.liquidity_score) * 0.002
        
        predicted_slippage = base_slippage + size_impact + volatility_impact + liquidity_impact
        
        historical = self.historical_slippage.get(symbol, [])
        if len(historical) >= 5:
            similar = [s for size, _, s in historical 
                      if 0.5 * order_size <= size <= 2 * order_size]
            if similar:
                avg_historical = sum(similar) / len(similar)
                predicted_slippage = predicted_slippage * 0.7 + avg_historical * 0.3
        
        return min(predicted_slippage, 0.05)
    
    def record_slippage(self, symbol: str, order_size: float, 
                       expected_price: float, actual_price: float) -> None:
        actual_slippage = abs(actual_price - expected_price) / expected_price if expected_price > 0 else 0
        
        if symbol not in self.historical_slippage:
            self.historical_slippage[symbol] = []
        
        self.historical_slippage[symbol].append((order_size, expected_price, actual_slippage))
        
        if len(self.historical_slippage[symbol]) > 100:
            self.historical_slippage[symbol] = self.historical_slippage[symbol][-100:]


class SmartExecutionEngine:
    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()
        self.slippage_predictor = SlippagePredictor()
        self.active_executions: Dict[str, ExecutionPlan] = {}
        self.execution_history: List[ExecutionResult] = []
        
    def analyze_market_conditions(self, order_book: Dict[str, Any],
                                  volume_data: Dict[str, float],
                                  volatility: float) -> MarketConditions:
        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])
        
        best_bid = bids[0][0] if bids else 0
        best_ask = asks[0][0] if asks else 0
        
        spread = best_ask - best_bid if best_bid and best_ask else 0
        mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
        spread_pct = spread / mid_price if mid_price > 0 else 0.01
        
        bid_depth = sum(price * qty for price, qty in bids[:10]) if bids else 0
        ask_depth = sum(price * qty for price, qty in asks[:10]) if asks else 0
        
        volume_24h = volume_data.get("volume_24h", 0)
        recent_volume = volume_data.get("recent_volume", 0)
        
        total_depth = bid_depth + ask_depth
        liquidity_score = min(1.0, total_depth / 100000) * (1 - min(1.0, spread_pct / 0.02))
        
        return MarketConditions(
            spread=spread,
            spread_pct=spread_pct,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            volatility=volatility,
            volume_24h=volume_24h,
            recent_volume=recent_volume,
            liquidity_score=liquidity_score
        )
    
    def select_strategy(self, order_size: float, order_value: float,
                       market_conditions: MarketConditions,
                       urgency: float = 0.5) -> ExecutionStrategy:
        if order_value < 100:
            return ExecutionStrategy.MARKET
        
        if urgency > 0.8:
            return ExecutionStrategy.MARKET
        
        depth_ratio = order_value / market_conditions.bid_depth if market_conditions.bid_depth > 0 else 1.0
        
        if depth_ratio > 0.5:
            return ExecutionStrategy.ICEBERG
        
        if depth_ratio > 0.2:
            if market_conditions.volatility > 0.3:
                return ExecutionStrategy.TWAP
            else:
                return ExecutionStrategy.VWAP
        
        if depth_ratio > 0.05:
            return ExecutionStrategy.ADAPTIVE
        
        return ExecutionStrategy.LIMIT
    
    def create_twap_plan(self, symbol: str, side: str, total_size: float,
                        current_price: float, duration_seconds: int = 300) -> ExecutionPlan:
        slices = []
        slice_size = total_size / self.config.twap_slices
        interval = duration_seconds / self.config.twap_slices
        
        for i in range(self.config.twap_slices):
            slices.append(OrderSlice(
                sequence=i,
                size=slice_size,
                price=None,
                order_type="market"
            ))
        
        return ExecutionPlan(
            strategy=ExecutionStrategy.TWAP,
            total_size=total_size,
            side=side,
            symbol=symbol,
            slices=slices,
            target_price=current_price,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(seconds=duration_seconds)
        )
    
    def create_vwap_plan(self, symbol: str, side: str, total_size: float,
                        current_price: float, volume_profile: List[float]) -> ExecutionPlan:
        slices = []
        
        if not volume_profile:
            volume_profile = [1.0 / self.config.twap_slices] * self.config.twap_slices
        
        total_volume = sum(volume_profile)
        
        for i, vol in enumerate(volume_profile):
            slice_pct = vol / total_volume if total_volume > 0 else 1.0 / len(volume_profile)
            slice_size = total_size * slice_pct * self.config.vwap_participation_rate
            
            if slice_size * current_price >= self.config.min_slice_value:
                slices.append(OrderSlice(
                    sequence=i,
                    size=slice_size,
                    price=None,
                    order_type="limit"
                ))
        
        remaining = total_size - sum(s.size for s in slices)
        if remaining > 0 and slices:
            slices[-1].size += remaining
        
        return ExecutionPlan(
            strategy=ExecutionStrategy.VWAP,
            total_size=total_size,
            side=side,
            symbol=symbol,
            slices=slices,
            target_price=current_price
        )
    
    def create_iceberg_plan(self, symbol: str, side: str, total_size: float,
                           limit_price: float) -> ExecutionPlan:
        visible_size = total_size * self.config.iceberg_show_pct
        num_slices = int(1 / self.config.iceberg_show_pct)
        
        slices = []
        remaining = total_size
        
        for i in range(num_slices):
            slice_size = min(visible_size, remaining)
            if slice_size * limit_price >= self.config.min_slice_value:
                slices.append(OrderSlice(
                    sequence=i,
                    size=slice_size,
                    price=limit_price,
                    order_type="limit"
                ))
                remaining -= slice_size
        
        if remaining > 0 and slices:
            slices[-1].size += remaining
        
        return ExecutionPlan(
            strategy=ExecutionStrategy.ICEBERG,
            total_size=total_size,
            side=side,
            symbol=symbol,
            slices=slices,
            target_price=limit_price,
            price_limit=limit_price
        )
    
    def create_adaptive_plan(self, symbol: str, side: str, total_size: float,
                            current_price: float, market_conditions: MarketConditions) -> ExecutionPlan:
        slices = []
        
        if market_conditions.volatility > 0.4:
            num_slices = 20
        elif market_conditions.volatility > 0.2:
            num_slices = 10
        else:
            num_slices = 5
        
        base_slice_size = total_size / num_slices
        
        for i in range(num_slices):
            volatility_adjustment = 1.0 + (0.5 - market_conditions.volatility) * 0.5
            adjusted_size = base_slice_size * volatility_adjustment
            
            slices.append(OrderSlice(
                sequence=i,
                size=adjusted_size,
                price=None,
                order_type="adaptive"
            ))
        
        total_planned = sum(s.size for s in slices)
        if total_planned != total_size and slices:
            adjustment = total_size / total_planned
            for s in slices:
                s.size *= adjustment
        
        return ExecutionPlan(
            strategy=ExecutionStrategy.ADAPTIVE,
            total_size=total_size,
            side=side,
            symbol=symbol,
            slices=slices,
            target_price=current_price,
            estimated_slippage=self.slippage_predictor.predict_slippage(
                symbol, total_size, market_conditions
            )
        )
    
    async def execute_plan(self, plan: ExecutionPlan,
                          execute_order_fn: Callable,
                          get_price_fn: Callable) -> ExecutionResult:
        start_time = time.time()
        details = []
        total_filled = 0.0
        total_cost = 0.0
        slices_completed = 0
        slices_failed = 0
        
        plan.start_time = datetime.now()
        self.active_executions[plan.symbol] = plan
        
        try:
            for slice_order in plan.slices:
                slice_order.status = "executing"
                slice_order.created_at = datetime.now()
                
                for attempt in range(self.config.max_retries):
                    try:
                        current_price = await get_price_fn(plan.symbol)
                        
                        if slice_order.order_type == "market":
                            price = None
                        elif slice_order.order_type == "limit":
                            if plan.side == "buy":
                                price = current_price * (1 - self.config.max_slippage_pct / 2)
                            else:
                                price = current_price * (1 + self.config.max_slippage_pct / 2)
                        else:
                            price = current_price
                        
                        result = await execute_order_fn(
                            symbol=plan.symbol,
                            side=plan.side,
                            size=slice_order.size,
                            price=price,
                            order_type="limit" if price else "market"
                        )
                        
                        if result and result.get("filled"):
                            slice_order.status = "filled"
                            slice_order.filled_size = result.get("filled_size", slice_order.size)
                            slice_order.filled_price = result.get("filled_price", current_price)
                            slice_order.filled_at = datetime.now()
                            slice_order.order_id = result.get("order_id")
                            
                            total_filled += slice_order.filled_size
                            total_cost += slice_order.filled_size * slice_order.filled_price
                            slices_completed += 1
                            
                            details.append({
                                "sequence": slice_order.sequence,
                                "size": slice_order.filled_size,
                                "price": slice_order.filled_price,
                                "status": "filled"
                            })
                            
                            break
                        else:
                            if attempt < self.config.max_retries - 1:
                                await asyncio.sleep(self.config.retry_delay_seconds)
                            
                    except Exception as e:
                        if attempt < self.config.max_retries - 1:
                            await asyncio.sleep(self.config.retry_delay_seconds)
                        else:
                            slice_order.status = "failed"
                            slices_failed += 1
                            details.append({
                                "sequence": slice_order.sequence,
                                "size": slice_order.size,
                                "status": "failed",
                                "error": str(e)
                            })
                
                if plan.strategy == ExecutionStrategy.TWAP:
                    interval = (plan.end_time - plan.start_time).total_seconds() / len(plan.slices) if plan.end_time else self.config.twap_interval_seconds
                    await asyncio.sleep(interval)
                elif plan.strategy in [ExecutionStrategy.VWAP, ExecutionStrategy.ADAPTIVE]:
                    await asyncio.sleep(1)
                
        finally:
            if plan.symbol in self.active_executions:
                del self.active_executions[plan.symbol]
        
        execution_time = time.time() - start_time
        average_price = total_cost / total_filled if total_filled > 0 else 0
        slippage = abs(average_price - plan.target_price) / plan.target_price if plan.target_price > 0 else 0
        slippage_cost = slippage * total_cost
        
        result = ExecutionResult(
            success=total_filled >= plan.total_size * 0.95,
            total_filled=total_filled,
            average_price=average_price,
            total_cost=total_cost,
            slippage=slippage,
            slippage_cost=slippage_cost,
            execution_time_seconds=execution_time,
            slices_completed=slices_completed,
            slices_failed=slices_failed,
            details=details
        )
        
        self.execution_history.append(result)
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]
        
        return result
    
    def estimate_execution_cost(self, symbol: str, size: float, side: str,
                               market_conditions: MarketConditions) -> Dict[str, Any]:
        predicted_slippage = self.slippage_predictor.predict_slippage(
            symbol, size, market_conditions
        )
        
        mid_price = (market_conditions.spread / 2) + market_conditions.bid_depth / 10000
        
        spread_cost = market_conditions.spread_pct / 2
        
        impact_cost = predicted_slippage
        
        total_cost_pct = spread_cost + impact_cost
        
        return {
            "spread_cost_pct": spread_cost,
            "market_impact_pct": impact_cost,
            "total_cost_pct": total_cost_pct,
            "predicted_slippage": predicted_slippage,
            "recommended_strategy": self.select_strategy(
                size, size * mid_price, market_conditions
            ).value
        }
    
    def get_execution_stats(self) -> Dict[str, Any]:
        if not self.execution_history:
            return {"executions": 0}
        
        successful = [e for e in self.execution_history if e.success]
        
        avg_slippage = sum(e.slippage for e in self.execution_history) / len(self.execution_history)
        avg_execution_time = sum(e.execution_time_seconds for e in self.execution_history) / len(self.execution_history)
        
        return {
            "executions": len(self.execution_history),
            "success_rate": len(successful) / len(self.execution_history),
            "avg_slippage": avg_slippage,
            "avg_execution_time": avg_execution_time,
            "total_slippage_cost": sum(e.slippage_cost for e in self.execution_history)
        }
