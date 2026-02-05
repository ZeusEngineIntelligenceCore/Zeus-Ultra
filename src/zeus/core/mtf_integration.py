#!/usr/bin/env python3
"""
MTF Integration Helper - Builds TimeframeIndicators for MTF Fusion
"""

from typing import Dict, List, Optional, Any
from ..indicators.mtf_fusion import TimeframeIndicators
from ..indicators.math_kernel import MathKernel


async def build_mtf_indicators(
    exchange,
    symbol: str,
    base_prices: List[float],
    base_volumes: List[float],
    math_kernel: Optional[MathKernel] = None
) -> Dict[str, TimeframeIndicators]:
    mk = math_kernel or MathKernel()
    timeframe_data = {}
    
    timeframes = ["5m", "15m", "1h", "4h", "1d"]
    
    for tf in timeframes:
        try:
            ohlcv = await exchange.fetch_ohlcv(symbol, tf, 200)
            if not ohlcv or len(ohlcv) < 50:
                continue
            
            high = [c.high for c in ohlcv]
            low = [c.low for c in ohlcv]
            close = [c.close for c in ohlcv]
            volume = [c.volume for c in ohlcv]
            
            rsi = mk.rsi(close, 14)[-1] if len(close) >= 14 else 50
            macd_line, signal_line, histogram = mk.macd(close)
            ema_9 = mk.ema(close, 9)[-1]
            ema_21 = mk.ema(close, 21)[-1]
            ema_50 = mk.ema(close, 50)[-1] if len(close) >= 50 else ema_21
            sma_200 = mk.sma(close, 200)[-1] if len(close) >= 200 else ema_50
            bb_upper, bb_middle, bb_lower = mk.bollinger_bands(close)
            atr = mk.atr(high, low, close, 14)[-1]
            adx, plus_di, minus_di = mk.adx(high, low, close)
            stoch_k, stoch_d = mk.stochastic(high, low, close)
            cci = mk.cci(high, low, close)[-1]
            mfi = mk.mfi(high, low, close, volume)[-1] if len(volume) >= 14 else 50
            obv = mk.obv(close, volume)
            obv_trend = (obv[-1] - obv[-10]) / abs(obv[-10]) if len(obv) >= 10 and obv[-10] != 0 else 0
            vol_sma = mk.sma(volume, 20)[-1] if len(volume) >= 20 else volume[-1]
            
            timeframe_data[tf] = TimeframeIndicators(
                timeframe=tf,
                rsi=rsi,
                macd=macd_line[-1],
                macd_signal=signal_line[-1],
                macd_histogram=histogram[-1],
                ema_9=ema_9,
                ema_21=ema_21,
                ema_50=ema_50,
                sma_200=sma_200,
                bb_upper=bb_upper[-1],
                bb_middle=bb_middle[-1],
                bb_lower=bb_lower[-1],
                bb_width=bb_upper[-1] - bb_lower[-1],
                atr=atr,
                adx=adx[-1],
                plus_di=plus_di[-1],
                minus_di=minus_di[-1],
                stoch_k=stoch_k[-1],
                stoch_d=stoch_d[-1],
                cci=cci,
                mfi=mfi,
                obv_trend=obv_trend,
                vwap=close[-1],
                current_price=close[-1],
                volume=volume[-1],
                volume_sma=vol_sma
            )
        except Exception:
            continue
    
    return timeframe_data
