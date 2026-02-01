#!/usr/bin/env python3
"""
ZEUS ADVANCED RISK ENGINE
Institutional-grade risk management with:
- Dynamic Kelly Criterion with volatility adjustment
- Value at Risk (VaR) and Expected Shortfall (ES)
- Correlation matrix for portfolio risk
- Monte Carlo simulations for drawdown probability
- Circuit breakers and exposure limits
"""

from __future__ import annotations
import math
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np


@dataclass
class RiskConfig:
    max_position_size_pct: float = 0.10
    max_portfolio_risk_pct: float = 0.25
    max_single_trade_risk_pct: float = 0.02
    max_daily_loss_pct: float = 0.05
    max_drawdown_pct: float = 0.15
    var_confidence: float = 0.95
    es_confidence: float = 0.975
    lookback_periods: int = 100
    kelly_fraction: float = 0.25
    min_kelly: float = 0.01
    max_kelly: float = 0.20
    correlation_threshold: float = 0.7
    monte_carlo_simulations: int = 1000
    volatility_lookback: int = 20
    circuit_breaker_cooldown_minutes: int = 60


@dataclass
class PositionRisk:
    symbol: str
    size: float
    entry_price: float
    current_price: float
    pnl: float
    pnl_pct: float
    var_95: float
    var_99: float
    expected_shortfall: float
    volatility: float
    beta: float = 1.0
    correlation_risk: float = 0.0


@dataclass
class PortfolioRisk:
    total_exposure: float
    total_pnl: float
    total_pnl_pct: float
    portfolio_var_95: float
    portfolio_var_99: float
    portfolio_es: float
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    correlation_matrix: Dict[str, Dict[str, float]]
    concentrated_risk: float
    diversification_ratio: float
    position_risks: List[PositionRisk]
    risk_signals: List[str] = field(default_factory=list)
    circuit_breaker_active: bool = False


@dataclass
class TradeRiskAssessment:
    approved: bool
    recommended_size: float
    max_size: float
    kelly_size: float
    var_impact: float
    portfolio_correlation: float
    risk_reward_ratio: float
    rejection_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class AdvancedRiskEngine:
    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        self.return_history: Dict[str, List[float]] = {}
        self.price_history: Dict[str, List[float]] = {}
        self.portfolio_value_history: List[float] = []
        self.peak_portfolio_value: float = 0.0
        self.daily_pnl: float = 0.0
        self.daily_start_value: float = 0.0
        self.last_reset_date: Optional[datetime] = None
        self.circuit_breaker_until: Optional[datetime] = None
        self.trade_history: List[Dict[str, Any]] = []
        
    def update_price(self, symbol: str, price: float) -> None:
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append(price)
        
        if len(self.price_history[symbol]) > self.config.lookback_periods + 10:
            self.price_history[symbol] = self.price_history[symbol][-self.config.lookback_periods - 10:]
        
        if len(self.price_history[symbol]) >= 2:
            if symbol not in self.return_history:
                self.return_history[symbol] = []
            
            prev_price = self.price_history[symbol][-2]
            if prev_price > 0:
                ret = (price - prev_price) / prev_price
                self.return_history[symbol].append(ret)
                
                if len(self.return_history[symbol]) > self.config.lookback_periods:
                    self.return_history[symbol] = self.return_history[symbol][-self.config.lookback_periods:]
    
    def update_portfolio_value(self, value: float) -> None:
        self.portfolio_value_history.append(value)
        
        if len(self.portfolio_value_history) > 1000:
            self.portfolio_value_history = self.portfolio_value_history[-1000:]
        
        if value > self.peak_portfolio_value:
            self.peak_portfolio_value = value
        
        now = datetime.now()
        if self.last_reset_date is None or now.date() != self.last_reset_date.date():
            self.daily_start_value = value
            self.daily_pnl = 0.0
            self.last_reset_date = now
        else:
            self.daily_pnl = value - self.daily_start_value
    
    def calculate_volatility(self, symbol: str) -> float:
        returns = self.return_history.get(symbol, [])
        
        if len(returns) < 5:
            return 0.02
        
        recent_returns = returns[-self.config.volatility_lookback:]
        mean_return = sum(recent_returns) / len(recent_returns)
        variance = sum((r - mean_return) ** 2 for r in recent_returns) / len(recent_returns)
        daily_vol = math.sqrt(variance)
        
        annualized_vol = daily_vol * math.sqrt(365)
        
        return annualized_vol
    
    def calculate_var(self, returns: List[float], confidence: float = 0.95) -> float:
        if len(returns) < 10:
            return 0.05
        
        sorted_returns = sorted(returns)
        index = int((1 - confidence) * len(sorted_returns))
        var = -sorted_returns[max(0, index)]
        
        return max(0.0, var)
    
    def calculate_expected_shortfall(self, returns: List[float], confidence: float = 0.975) -> float:
        if len(returns) < 10:
            return 0.07
        
        sorted_returns = sorted(returns)
        cutoff_index = int((1 - confidence) * len(sorted_returns))
        
        if cutoff_index == 0:
            return -sorted_returns[0]
        
        tail_returns = sorted_returns[:cutoff_index]
        es = -sum(tail_returns) / len(tail_returns)
        
        return max(0.0, es)
    
    def calculate_correlation_matrix(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        matrix: Dict[str, Dict[str, float]] = {}
        
        for s1 in symbols:
            matrix[s1] = {}
            for s2 in symbols:
                if s1 == s2:
                    matrix[s1][s2] = 1.0
                elif s2 in matrix and s1 in matrix[s2]:
                    matrix[s1][s2] = matrix[s2][s1]
                else:
                    matrix[s1][s2] = self._calculate_correlation(s1, s2)
        
        return matrix
    
    def _calculate_correlation(self, symbol1: str, symbol2: str) -> float:
        returns1 = self.return_history.get(symbol1, [])
        returns2 = self.return_history.get(symbol2, [])
        
        min_len = min(len(returns1), len(returns2))
        
        if min_len < 10:
            return 0.5
        
        r1 = returns1[-min_len:]
        r2 = returns2[-min_len:]
        
        mean1 = sum(r1) / len(r1)
        mean2 = sum(r2) / len(r2)
        
        covariance = sum((r1[i] - mean1) * (r2[i] - mean2) for i in range(min_len)) / min_len
        std1 = math.sqrt(sum((r - mean1) ** 2 for r in r1) / len(r1))
        std2 = math.sqrt(sum((r - mean2) ** 2 for r in r2) / len(r2))
        
        if std1 == 0 or std2 == 0:
            return 0.0
        
        correlation = covariance / (std1 * std2)
        
        return max(-1.0, min(1.0, correlation))
    
    def calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float,
                                  volatility: float = 0.0) -> float:
        if avg_loss == 0:
            return self.config.min_kelly
        
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 1.0
        
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        kelly *= self.config.kelly_fraction
        
        if volatility > 0.5:
            volatility_adjustment = 1.0 - min(0.5, (volatility - 0.5))
            kelly *= volatility_adjustment
        
        return max(self.config.min_kelly, min(self.config.max_kelly, kelly))
    
    def monte_carlo_drawdown(self, returns: List[float]) -> Tuple[float, float]:
        if len(returns) < 20:
            return 0.1, 0.2
        
        mean_return = sum(returns) / len(returns)
        std_return = math.sqrt(sum((r - mean_return) ** 2 for r in returns) / len(returns))
        
        max_drawdowns = []
        
        for _ in range(self.config.monte_carlo_simulations):
            simulated_returns = [random.gauss(mean_return, std_return) for _ in range(252)]
            
            cumulative = 1.0
            peak = 1.0
            max_dd = 0.0
            
            for ret in simulated_returns:
                cumulative *= (1 + ret)
                if cumulative > peak:
                    peak = cumulative
                drawdown = (peak - cumulative) / peak if peak > 0 else 0
                max_dd = max(max_dd, drawdown)
            
            max_drawdowns.append(max_dd)
        
        sorted_dd = sorted(max_drawdowns)
        expected_dd = sum(sorted_dd) / len(sorted_dd)
        worst_case_dd = sorted_dd[int(0.95 * len(sorted_dd))]
        
        return expected_dd, worst_case_dd
    
    def calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.05) -> float:
        if len(returns) < 10:
            return 0.0
        
        mean_return = sum(returns) / len(returns)
        std_return = math.sqrt(sum((r - mean_return) ** 2 for r in returns) / len(returns))
        
        if std_return == 0:
            return 0.0
        
        annual_return = mean_return * 365
        annual_std = std_return * math.sqrt(365)
        
        sharpe = (annual_return - risk_free_rate) / annual_std
        
        return sharpe
    
    def calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float = 0.05) -> float:
        if len(returns) < 10:
            return 0.0
        
        mean_return = sum(returns) / len(returns)
        
        negative_returns = [r for r in returns if r < 0]
        if not negative_returns:
            return 10.0
        
        downside_deviation = math.sqrt(sum(r ** 2 for r in negative_returns) / len(negative_returns))
        
        if downside_deviation == 0:
            return 0.0
        
        annual_return = mean_return * 365
        annual_downside = downside_deviation * math.sqrt(365)
        
        sortino = (annual_return - risk_free_rate) / annual_downside
        
        return sortino
    
    def assess_portfolio_risk(self, positions: List[Dict[str, Any]], 
                             portfolio_value: float) -> PortfolioRisk:
        self.update_portfolio_value(portfolio_value)
        
        position_risks = []
        symbols = []
        
        total_exposure = 0.0
        total_pnl = 0.0
        
        for pos in positions:
            symbol = pos.get("symbol", "")
            size = pos.get("size", 0)
            entry_price = pos.get("entry_price", 0)
            current_price = pos.get("current_price", entry_price)
            
            symbols.append(symbol)
            
            pnl = (current_price - entry_price) * size
            pnl_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0
            
            exposure = size * current_price
            total_exposure += exposure
            total_pnl += pnl
            
            returns = self.return_history.get(symbol, [])
            volatility = self.calculate_volatility(symbol)
            var_95 = self.calculate_var(returns, 0.95) * exposure
            var_99 = self.calculate_var(returns, 0.99) * exposure
            es = self.calculate_expected_shortfall(returns) * exposure
            
            position_risks.append(PositionRisk(
                symbol=symbol,
                size=size,
                entry_price=entry_price,
                current_price=current_price,
                pnl=pnl,
                pnl_pct=pnl_pct,
                var_95=var_95,
                var_99=var_99,
                expected_shortfall=es,
                volatility=volatility
            ))
        
        correlation_matrix = self.calculate_correlation_matrix(symbols) if len(symbols) > 1 else {}
        
        portfolio_var_95 = sum(p.var_95 for p in position_risks)
        portfolio_var_99 = sum(p.var_99 for p in position_risks)
        portfolio_es = sum(p.expected_shortfall for p in position_risks)
        
        if len(symbols) > 1 and correlation_matrix:
            diversification_benefit = 0.0
            for i, s1 in enumerate(symbols):
                for j, s2 in enumerate(symbols):
                    if i < j:
                        corr = correlation_matrix.get(s1, {}).get(s2, 0.5)
                        if corr < 0.5:
                            diversification_benefit += (0.5 - corr) * 0.1
            
            portfolio_var_95 *= (1 - min(0.3, diversification_benefit))
            portfolio_var_99 *= (1 - min(0.3, diversification_benefit))
        
        current_drawdown = 0.0
        if self.peak_portfolio_value > 0:
            current_drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
        
        max_drawdown = current_drawdown
        if len(self.portfolio_value_history) > 10:
            peak = self.portfolio_value_history[0]
            for value in self.portfolio_value_history:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak if peak > 0 else 0
                max_drawdown = max(max_drawdown, dd)
        
        all_returns = []
        for returns in self.return_history.values():
            all_returns.extend(returns[-50:])
        
        sharpe = self.calculate_sharpe_ratio(all_returns)
        sortino = self.calculate_sortino_ratio(all_returns)
        calmar = (sum(all_returns) / len(all_returns) * 365) / max_drawdown if max_drawdown > 0 and all_returns else 0
        
        exposures = [p.size * p.current_price for p in position_risks]
        total_exp = sum(exposures)
        concentrated_risk = max(exposures) / total_exp if total_exp > 0 and exposures else 0
        
        individual_vol = sum(p.volatility for p in position_risks)
        portfolio_vol = math.sqrt(sum(p.volatility ** 2 for p in position_risks))
        diversification_ratio = individual_vol / portfolio_vol if portfolio_vol > 0 else 1.0
        
        risk_signals = []
        circuit_breaker = False
        
        if current_drawdown > self.config.max_drawdown_pct:
            risk_signals.append(f"CRITICAL: Drawdown {current_drawdown:.1%} exceeds limit {self.config.max_drawdown_pct:.1%}")
            circuit_breaker = True
        
        if portfolio_value > 0:
            daily_loss_pct = -self.daily_pnl / portfolio_value if self.daily_pnl < 0 else 0
            if daily_loss_pct > self.config.max_daily_loss_pct:
                risk_signals.append(f"CRITICAL: Daily loss {daily_loss_pct:.1%} exceeds limit {self.config.max_daily_loss_pct:.1%}")
                circuit_breaker = True
        
        if portfolio_value > 0 and total_exposure / portfolio_value > self.config.max_portfolio_risk_pct * 5:
            risk_signals.append(f"WARNING: High portfolio exposure {total_exposure/portfolio_value:.1%}")
        
        if concentrated_risk > 0.4:
            risk_signals.append(f"WARNING: Concentrated position risk {concentrated_risk:.1%}")
        
        if circuit_breaker:
            self.circuit_breaker_until = datetime.now() + timedelta(minutes=self.config.circuit_breaker_cooldown_minutes)
        
        return PortfolioRisk(
            total_exposure=total_exposure,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl / portfolio_value if portfolio_value > 0 else 0,
            portfolio_var_95=portfolio_var_95,
            portfolio_var_99=portfolio_var_99,
            portfolio_es=portfolio_es,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            correlation_matrix=correlation_matrix,
            concentrated_risk=concentrated_risk,
            diversification_ratio=diversification_ratio,
            position_risks=position_risks,
            risk_signals=risk_signals,
            circuit_breaker_active=circuit_breaker
        )
    
    def assess_trade_risk(self, symbol: str, proposed_size: float, entry_price: float,
                         stop_loss: float, take_profit: float, portfolio_value: float,
                         current_positions: List[Dict[str, Any]]) -> TradeRiskAssessment:
        rejection_reasons = []
        warnings = []
        
        if self.circuit_breaker_until and datetime.now() < self.circuit_breaker_until:
            return TradeRiskAssessment(
                approved=False,
                recommended_size=0,
                max_size=0,
                kelly_size=0,
                var_impact=0,
                portfolio_correlation=0,
                risk_reward_ratio=0,
                rejection_reasons=["Circuit breaker active - trading paused"]
            )
        
        trade_value = proposed_size * entry_price
        
        if portfolio_value > 0 and trade_value / portfolio_value > self.config.max_position_size_pct:
            rejection_reasons.append(f"Position size {trade_value/portfolio_value:.1%} exceeds max {self.config.max_position_size_pct:.1%}")
        
        trade_risk = abs(entry_price - stop_loss) / entry_price if entry_price > 0 else 0
        trade_risk_value = trade_risk * trade_value
        
        if portfolio_value > 0 and trade_risk_value / portfolio_value > self.config.max_single_trade_risk_pct:
            rejection_reasons.append(f"Trade risk {trade_risk_value/portfolio_value:.1%} exceeds max {self.config.max_single_trade_risk_pct:.1%}")
        
        potential_reward = abs(take_profit - entry_price) / entry_price if entry_price > 0 else 0
        risk_reward_ratio = potential_reward / trade_risk if trade_risk > 0 else 0
        
        if risk_reward_ratio < 1.5:
            warnings.append(f"Low risk/reward ratio: {risk_reward_ratio:.2f}")
        
        volatility = self.calculate_volatility(symbol)
        returns = self.return_history.get(symbol, [])
        var_impact = self.calculate_var(returns) * trade_value
        
        portfolio_correlation = 0.0
        if current_positions:
            correlations = []
            for pos in current_positions:
                pos_symbol = pos.get("symbol", "")
                if pos_symbol and pos_symbol != symbol:
                    corr = self._calculate_correlation(symbol, pos_symbol)
                    correlations.append(abs(corr))
            
            if correlations:
                portfolio_correlation = sum(correlations) / len(correlations)
                
                if portfolio_correlation > self.config.correlation_threshold:
                    warnings.append(f"High correlation with existing positions: {portfolio_correlation:.2f}")
        
        win_rate = 0.55
        avg_win = potential_reward
        avg_loss = trade_risk
        
        kelly_fraction = self.calculate_kelly_criterion(win_rate, avg_win, avg_loss, volatility)
        kelly_size = kelly_fraction * portfolio_value / entry_price if entry_price > 0 else 0
        
        max_size_by_position = self.config.max_position_size_pct * portfolio_value / entry_price if entry_price > 0 else 0
        max_size_by_risk = self.config.max_single_trade_risk_pct * portfolio_value / (trade_risk * entry_price) if trade_risk > 0 and entry_price > 0 else 0
        
        max_size = min(max_size_by_position, max_size_by_risk, kelly_size * 2)
        recommended_size = min(proposed_size, kelly_size, max_size)
        
        approved = len(rejection_reasons) == 0 and recommended_size > 0
        
        return TradeRiskAssessment(
            approved=approved,
            recommended_size=recommended_size,
            max_size=max_size,
            kelly_size=kelly_size,
            var_impact=var_impact,
            portfolio_correlation=portfolio_correlation,
            risk_reward_ratio=risk_reward_ratio,
            rejection_reasons=rejection_reasons,
            warnings=warnings
        )
    
    def record_trade_outcome(self, symbol: str, pnl: float, pnl_pct: float,
                            hold_duration: float) -> None:
        self.trade_history.append({
            "symbol": symbol,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "hold_duration": hold_duration,
            "timestamp": datetime.now().isoformat()
        })
        
        if len(self.trade_history) > 500:
            self.trade_history = self.trade_history[-500:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        if not self.trade_history:
            return {"trades": 0, "win_rate": 0, "avg_pnl": 0}
        
        wins = [t for t in self.trade_history if t["pnl"] > 0]
        losses = [t for t in self.trade_history if t["pnl"] <= 0]
        
        win_rate = len(wins) / len(self.trade_history) if self.trade_history else 0
        avg_pnl = sum(t["pnl"] for t in self.trade_history) / len(self.trade_history)
        avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t["pnl"] for t in losses) / len(losses) if losses else 0
        
        return {
            "trades": len(self.trade_history),
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": abs(sum(t["pnl"] for t in wins) / sum(t["pnl"] for t in losses)) if losses and sum(t["pnl"] for t in losses) != 0 else 0
        }
