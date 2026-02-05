#!/usr/bin/env python3
"""
ZEUS SECRET WEAPONS - Advanced Market Intelligence
Smart Money Flow, Whale Detection, Altcoin Season, Portfolio Correlation
"""

from __future__ import annotations
import math
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path

logger = logging.getLogger("Zeus.SecretWeapons")


@dataclass
class SmartMoneySignal:
    direction: str  # "ACCUMULATION", "DISTRIBUTION", "NEUTRAL"
    strength: float  # 0-100
    confidence: float  # 0-1
    indicators: Dict[str, float] = field(default_factory=dict)
    timestamp: str = ""


@dataclass
class WhaleActivity:
    activity_level: str  # "HIGH", "MODERATE", "LOW"
    direction: str  # "BUYING", "SELLING", "MIXED"
    large_order_ratio: float
    volume_anomaly: float
    detected_at: str = ""


@dataclass
class AltcoinSeasonMetrics:
    is_altseason: bool
    btc_dominance: float
    alt_performance_vs_btc: float
    season_score: float  # 0-100, >75 = altseason
    top_performers: List[str] = field(default_factory=list)


class SecretWeapons:
    def __init__(self, data_file: str = "data/secret_weapons.json"):
        self.data_file = Path(data_file)
        self.smart_money_history: List[Dict] = []
        self.whale_history: List[Dict] = []
        self.altseason_history: List[Dict] = []
        self._load_state()
    
    def _load_state(self) -> None:
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.smart_money_history = data.get("smart_money_history", [])[-100:]
                    self.whale_history = data.get("whale_history", [])[-100:]
                    self.altseason_history = data.get("altseason_history", [])[-100:]
            except Exception as e:
                logger.error(f"Failed to load secret weapons state: {e}")
    
    def _save_state(self) -> None:
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = {
                "smart_money_history": self.smart_money_history[-100:],
                "whale_history": self.whale_history[-100:],
                "altseason_history": self.altseason_history[-100:],
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save secret weapons state: {e}")

    def detect_smart_money_flow(
        self,
        close: List[float],
        volume: List[float],
        high: List[float],
        low: List[float]
    ) -> SmartMoneySignal:
        if len(close) < 50:
            return SmartMoneySignal("NEUTRAL", 0, 0, {}, datetime.now(timezone.utc).isoformat())
        
        n = len(close)
        
        obv = 0.0
        obv_values = [0.0]
        for i in range(1, n):
            if close[i] > close[i-1]:
                obv += volume[i]
            elif close[i] < close[i-1]:
                obv -= volume[i]
            obv_values.append(obv)
        
        obv_slope = (obv_values[-1] - obv_values[-20]) / 20 if len(obv_values) >= 20 else 0
        price_slope = (close[-1] - close[-20]) / close[-20] * 100 if len(close) >= 20 else 0
        
        divergence = 0.0
        if price_slope < 0 and obv_slope > 0:
            divergence = min(100, abs(obv_slope) / (abs(volume[-1]) / 100 + 1))
        elif price_slope > 0 and obv_slope < 0:
            divergence = -min(100, abs(obv_slope) / (abs(volume[-1]) / 100 + 1))
        
        cmf_values = []
        for i in range(-20, 0):
            if high[i] - low[i] > 0:
                mfm = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
                mfv = mfm * volume[i]
                cmf_values.append(mfv)
        cmf = sum(cmf_values) / (sum(volume[-20:]) + 1e-10)
        
        avg_vol = sum(volume[-20:]) / 20
        vol_trend = (sum(volume[-5:]) / 5 - avg_vol) / (avg_vol + 1e-10)
        
        up_volume = sum(volume[i] for i in range(-20, 0) if close[i] > close[i-1])
        down_volume = sum(volume[i] for i in range(-20, 0) if close[i] < close[i-1])
        ad_ratio = up_volume / (down_volume + 1e-10)
        
        smart_money_score = 50.0
        if cmf > 0.1:
            smart_money_score += 15
        elif cmf < -0.1:
            smart_money_score -= 15
        if divergence > 20:
            smart_money_score += 20
        elif divergence < -20:
            smart_money_score -= 20
        if ad_ratio > 1.5:
            smart_money_score += 10
        elif ad_ratio < 0.67:
            smart_money_score -= 10
        if vol_trend > 0.5:
            smart_money_score += 5
        
        smart_money_score = max(0, min(100, smart_money_score))
        
        if smart_money_score >= 65:
            direction = "ACCUMULATION"
        elif smart_money_score <= 35:
            direction = "DISTRIBUTION"
        else:
            direction = "NEUTRAL"
        
        confidence = abs(smart_money_score - 50) / 50
        
        signal = SmartMoneySignal(
            direction=direction,
            strength=smart_money_score,
            confidence=confidence,
            indicators={
                "cmf": round(cmf, 4),
                "obv_slope": round(obv_slope, 4),
                "divergence": round(divergence, 2),
                "ad_ratio": round(ad_ratio, 2),
                "vol_trend": round(vol_trend, 2)
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        self.smart_money_history.append({
            "direction": direction,
            "strength": smart_money_score,
            "timestamp": signal.timestamp
        })
        self._save_state()
        
        return signal

    def detect_whale_activity(
        self,
        volume: List[float],
        close: List[float],
        trade_sizes: Optional[List[float]] = None
    ) -> WhaleActivity:
        if len(volume) < 30:
            return WhaleActivity("LOW", "MIXED", 0, 0, datetime.now(timezone.utc).isoformat())
        
        avg_volume = sum(volume[-30:]) / 30
        std_volume = math.sqrt(sum((v - avg_volume) ** 2 for v in volume[-30:]) / 30)
        
        recent_volume = sum(volume[-5:]) / 5
        volume_anomaly = (recent_volume - avg_volume) / (std_volume + 1e-10)
        
        volume_spikes = sum(1 for v in volume[-10:] if v > avg_volume * 2)
        spike_ratio = volume_spikes / 10
        
        large_order_ratio = spike_ratio
        if trade_sizes:
            avg_size = sum(trade_sizes) / len(trade_sizes)
            large_orders = sum(1 for s in trade_sizes[-50:] if s > avg_size * 3)
            large_order_ratio = large_orders / min(50, len(trade_sizes))
        
        up_moves = sum(1 for i in range(-10, 0) if close[i] > close[i-1])
        direction = "BUYING" if up_moves >= 7 else "SELLING" if up_moves <= 3 else "MIXED"
        
        activity_score = (volume_anomaly + spike_ratio * 3 + large_order_ratio * 3) / 3
        activity_level = "HIGH" if activity_score > 1.5 else "MODERATE" if activity_score > 0.5 else "LOW"
        
        activity = WhaleActivity(
            activity_level=activity_level,
            direction=direction,
            large_order_ratio=round(large_order_ratio, 3),
            volume_anomaly=round(volume_anomaly, 2),
            detected_at=datetime.now(timezone.utc).isoformat()
        )
        
        self.whale_history.append({
            "activity_level": activity_level,
            "direction": direction,
            "volume_anomaly": volume_anomaly,
            "timestamp": activity.detected_at
        })
        self._save_state()
        
        return activity

    def calculate_altcoin_season(
        self,
        btc_return: float,
        alt_returns: Dict[str, float],
        btc_dominance: Optional[float] = None
    ) -> AltcoinSeasonMetrics:
        if not alt_returns:
            return AltcoinSeasonMetrics(False, btc_dominance or 50, 0, 0, [])
        
        outperforming = sum(1 for ret in alt_returns.values() if ret > btc_return)
        outperform_ratio = outperforming / len(alt_returns)
        
        avg_alt_return = sum(alt_returns.values()) / len(alt_returns)
        alt_vs_btc = avg_alt_return - btc_return
        
        season_score = 50.0
        if outperform_ratio > 0.75:
            season_score += 25
        elif outperform_ratio > 0.5:
            season_score += 10
        elif outperform_ratio < 0.25:
            season_score -= 25
        
        if alt_vs_btc > 10:
            season_score += 20
        elif alt_vs_btc > 5:
            season_score += 10
        elif alt_vs_btc < -10:
            season_score -= 20
        
        if btc_dominance:
            if btc_dominance < 40:
                season_score += 15
            elif btc_dominance < 45:
                season_score += 10
            elif btc_dominance > 55:
                season_score -= 15
        
        season_score = max(0, min(100, season_score))
        is_altseason = season_score >= 75
        
        sorted_alts = sorted(alt_returns.items(), key=lambda x: x[1], reverse=True)
        top_performers = [symbol for symbol, _ in sorted_alts[:5]]
        
        metrics = AltcoinSeasonMetrics(
            is_altseason=is_altseason,
            btc_dominance=btc_dominance or 50,
            alt_performance_vs_btc=round(alt_vs_btc, 2),
            season_score=round(season_score, 1),
            top_performers=top_performers
        )
        
        self.altseason_history.append({
            "is_altseason": is_altseason,
            "season_score": season_score,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        self._save_state()
        
        return metrics

    def calculate_portfolio_correlation(
        self,
        holdings: Dict[str, float],
        price_histories: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        if len(holdings) < 2 or len(price_histories) < 2:
            return {"status": "insufficient_data", "correlations": {}, "diversification_score": 0}
        
        returns_map = {}
        for symbol, prices in price_histories.items():
            if len(prices) >= 20:
                returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
                returns_map[symbol] = returns[-20:]
        
        if len(returns_map) < 2:
            return {"status": "insufficient_data", "correlations": {}, "diversification_score": 0}
        
        correlations = {}
        symbols = list(returns_map.keys())
        
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
                r1 = returns_map[sym1]
                r2 = returns_map[sym2]
                min_len = min(len(r1), len(r2))
                if min_len < 10:
                    continue
                r1, r2 = r1[-min_len:], r2[-min_len:]
                
                mean1, mean2 = sum(r1)/len(r1), sum(r2)/len(r2)
                cov = sum((r1[j] - mean1) * (r2[j] - mean2) for j in range(min_len)) / min_len
                std1 = math.sqrt(sum((x - mean1)**2 for x in r1) / len(r1))
                std2 = math.sqrt(sum((x - mean2)**2 for x in r2) / len(r2))
                
                if std1 > 0 and std2 > 0:
                    corr = cov / (std1 * std2)
                    correlations[f"{sym1}_{sym2}"] = round(corr, 3)
        
        avg_corr = sum(abs(c) for c in correlations.values()) / len(correlations) if correlations else 0
        diversification_score = max(0, min(100, (1 - avg_corr) * 100))
        
        high_corr_pairs = [(pair, c) for pair, c in correlations.items() if abs(c) > 0.7]
        
        return {
            "status": "success",
            "correlations": correlations,
            "average_correlation": round(avg_corr, 3),
            "diversification_score": round(diversification_score, 1),
            "high_correlation_pairs": high_corr_pairs,
            "recommendation": "Well diversified" if diversification_score > 70 else "Consider diversifying" if diversification_score > 40 else "High correlation risk"
        }

    def get_summary(self) -> Dict[str, Any]:
        latest_smart_money = self.smart_money_history[-1] if self.smart_money_history else None
        latest_whale = self.whale_history[-1] if self.whale_history else None
        latest_altseason = self.altseason_history[-1] if self.altseason_history else None
        
        return {
            "smart_money": latest_smart_money,
            "whale_activity": latest_whale,
            "altcoin_season": latest_altseason,
            "history_counts": {
                "smart_money": len(self.smart_money_history),
                "whale": len(self.whale_history),
                "altseason": len(self.altseason_history)
            }
        }
