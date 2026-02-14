# Zeus Autonomous Trading Bot

## Overview
Zeus is a professional-grade autonomous cryptocurrency trading bot designed for the Kraken exchange with a multi-exchange architecture. Its core purpose is to identify and capitalize on pre-breakout trading opportunities using advanced technical analysis. Key capabilities include sophisticated market microstructure analysis, institutional-grade risk management, smart order execution, and AI/ML-driven learning. The project aims to provide a robust, self-improving trading solution for cryptocurrency markets, offering capabilities typically reserved for institutional traders to a broader audience.

## User Preferences
- The bot should NOT auto-start. It must be manually toggled ON from the dashboard.
- In "LIVE" mode, the bot should execute trades.
- In "PAPER" mode, the bot should allow safe testing without live trading.
- Telegram alerts should be sent, adapting to trading preferences and patterns.
- The bot should NEVER auto-close manually placed orders, preserving full user control.
- The bot should NOT sell at a loss; it should wait for recovery.
- Non-urgent Telegram alerts should be batched and summarized every 30 minutes.

## System Architecture
Zeus is built around a modular architecture with a core trading engine supported by several specialized modules.

### UI/UX Decisions
- An interactive dashboard provides real-time monitoring of active trades, top candidates, and bot status.
- The dashboard includes a bot toggle on/off switch.
- Telegram Mini App and dashboard commands offer interactive coin analysis and direct access to the web dashboard.

### Technical Implementations
- **Core Trading Engine**: Utilizes over 50 technical indicators and a 30 KPI pre-breakout scoring system with weighted multi-factor analysis. It focuses on the top 20 priority candidates and performs multi-timeframe analysis (5m, 15m, 1h, 4h, 1d) with 500 candles each. Includes detailed [TRADE BLOCKED]/[TRADE APPROVED] logging for full visibility into trade decisions.
- **Advanced Market Microstructure**: Incorporates Order Flow Imbalance (OFI), VPIN Toxicity Detection, Iceberg Order Detection, Market Maker Bias Analysis, and Liquidity Scoring.
- **Institutional-Grade Risk Management**: Features Dynamic Kelly Criterion, Value at Risk (VaR), Expected Shortfall (ES), Correlation Matrix, Circuit Breakers, and Monte Carlo Simulations.
- **Smart Order Execution**: Implements TWAP, VWAP, and Iceberg order algorithms, along with ML-based slippage prediction and adaptive execution strategies.
- **Market Regime Detection**: Uses Hidden Markov Models (HMM) and GARCH for volatility clustering, enabling automatic strategy adaptation based on trend vs. range and momentum vs. mean-reversion classifications.
- **Advanced Mathematical Analysis**: Applies Hurst Exponent, Fractal Support/Resistance, Wavelet Denoising, Shannon Entropy, and Information Ratio for signal quality assessment.
- **AI/ML Learning System**: Features a triple-layer learning architecture:
  1. **Basic ML Engine**: Tracks performance by symbol, strategy, and time of day. Adapts confidence thresholds based on historical win rates.
  2. **Advanced ML Engine** (NEW): Implements reinforcement learning with Q-value optimization, pattern similarity matching using feature vectors (21 normalized features), market regime detection (TRENDING_UP/DOWN, RANGING, HIGH/LOW_VOLATILITY), adaptive position sizing based on historical pattern success, and indicator importance ranking.
  3. **Telegram Learning Engine**: Adapts notifications based on user interaction patterns.
  The system continuously improves through optimization cycles that update indicator importance weights and confidence thresholds.
- **Secret Weapons Module** (NEW): Includes Smart Money Flow detection (OBV divergence, CMF, Accumulation/Distribution), Whale Activity detection (volume anomalies, large order ratio), Altcoin Season Index, and Portfolio Correlation Matrix for diversification analysis.
- **Advanced Exit Strategies** (NEW): Quick win detection (1.5% in 15 mins), partial profit taking at multiple levels, dynamic trailing stops, momentum reversal exits, and time-based exit optimization.
- **Breakout & Fakeout Analyzer** (ENHANCED): Enhanced breakout quality scoring, multi-factor fakeout detection, volume spike/fallout monitoring, whale activity signals, and integrated into the candidate pipeline to filter post-breakout exhaustion and fakeouts.
- **True Pre-Breakout Detection** (ENHANCED v2): 49-KPI scoring system with consolidation/accumulation focus. Core signals: Range Contraction, Higher Lows, OBV Divergence, Volume Accumulation, Resistance Proximity, Volume Profile (VPVR - low/high volume node analysis), Seller Exhaustion Detection (declining sell volume while price holds), Absorption Detection (support level defense by large buyers), Relative Volume (vs historical time-of-day patterns), Real Order Book Imbalance (live Kraken Level 2 data). Anti-extension filters penalize coins already extended (RSI overbought, >10% recent move). Weight hierarchy: consolidation signals (5-6.5%) > microstructure signals (3.5-5%) > traditional indicators (1%). Combo bonuses when multiple accumulation signals align (up to 22% score boost). PRE_BREAKOUT stage prioritized over BREAKOUT in scoring.
- **Predictive Limit Order Engine** (NEW): Replaces market buys with intelligent limit orders by:
  - Identifying support levels from price history
  - Detecting momentum exhaustion for reversal prediction
  - Analyzing order book for bid walls and smart money accumulation
  - Predicting optimal limit entry prices to get better fills
  - Tracking prediction accuracy for continuous improvement
- **Multi-Timeframe Fusion System** (NEW): Ultra-accurate trading signals by combining ALL indicators across 5 timeframes (5m/15m/1h/4h/1d):
  - Scores each timeframe on trend, momentum, volatility, and volume
  - Calculates consensus with alignment scoring (detects when all timeframes agree)
  - Detects divergences between short-term and long-term signals
  - Produces FusedSignal with direction, strength, confidence, and letter grade (A+ to D)
  - Boosts trade confidence when MTF alignment is high, reduces when conflicting
  - "Beats Competition Score" metric for signal quality comparison
- **21 Telegram Commands** (ENHANCED): Full command processing with polling:
  - /status, /portfolio, /trades, /candidates, /performance
  - /kpi, /regime, /ml, /fear, /scan
  - /best, /worst, /streak, /whale, /mtf
  - /analyze, /dashboard, /report, /settings, /help
- **KPI Analytics System**: Tracks 20+ critical performance metrics including Sharpe ratio, Sortino ratio, max drawdown, win rate, profit factor, expectancy, Calmar ratio, trade frequency, average hold times, symbol-level performance, Recovery Factor, Ulcer Index, Tail Ratio, CAGR, K-Ratio, Consecutive Streaks, Risk of Ruin, Common Sense Ratio, Gain-to-Pain Ratio, and Time-Weighted Return. Records daily returns and hourly performance analysis.
- **Trading Configuration**: Supports a maximum of 100 open positions, a minimum profit target of 3-4%, continuous holdings monitoring, full market scans every 300 seconds, and position scans every 60 seconds.
- **Authentication**: Uses Replit Auth for user authentication, supporting various social logins and email/password, with protected routes.
- **Production Server**: Employs a Gunicorn WSGI production server with 1 worker and 4 threads.
- **Timezone**: All timestamps are configured for Los Angeles timezone (America/Los_Angeles).
- **State Persistence**: Bot state (equity, holdings, trades) is persisted to `data/bot_state.json` and used as fallback for API responses when the in-memory bot instance isn't directly accessible (gunicorn process isolation).

### Feature Specifications
- **Pre-Breakout Detection**: 30 KPI scoring system for high-precision signal generation.
- **Trade Execution**: Immediate sell order placement upon buy order fill, with dynamic adjustment based on market conditions.
- **Continuous Trading**: Designed for day trading with continuous market and position scanning.
- **Alerts**: Real-time Telegram alerts for trade open/close notifications, with an adaptive learning engine.

## Recent Changes
- **2026-02-14**: Replaced hardcoded 4.5% take profit with professional trade level engine using swing high resistance, Fibonacci extensions (1.0/1.272/1.618), Bollinger Band upper, pivot points (R1/R2/R3 from 20-bar range), ATR-scaled targets, and weighted composite averaging - each coin gets unique TP based on its own price structure
- **2026-02-14**: Stop loss now derived from nearest swing low support levels (placed 0.3 ATR below support) instead of flat 2x ATR
- **2026-02-14**: Trade levels display shows methodology (resistance/fibonacci/pivot/atr_volatility/composite), risk/reward ratio with color coding, nearest resistance and support levels
- **2026-02-14**: TP capped at 15% max to prevent unrealistic targets; minimum 3% absolute floor; risk/reward enforcement blended (40/60) to avoid dominating composite
- **2026-02-14**: Eliminated pre-breakout score inflation - combo bonuses changed from multiplicative stacking (up to 7x) to additive with 1.30x cap
- **2026-02-14**: Confidence formula rewritten as independent directional measure (bullish agreement, bearish drag, dispersion penalty) instead of linear derivative of score
- **2026-02-14**: Strengthened anti-extension filters - RSI>75 now 0.45x penalty (was 0.75x), recent moves >15% now 0.40x (was 0.65x)
- **2026-02-14**: Fixed indicator biases in squeeze_pressure, donchian_position, cci_signal, williams_r_signal, stoch_rsi_signal - oversold no longer inflates pre-breakout scores
- **2026-02-14**: Added bearish majority check - if >50% indicators below 0.4, score gets 0.60x penalty
- **2026-02-10**: Fixed critical asyncio event loop conflict - coin analyzer and Telegram analysis endpoints now always create standalone exchange instances to avoid cross-thread Lock errors when the bot runs in a background thread
- **2026-02-10**: Telegram analysis send now uses direct HTTP requests to Telegram API instead of bot's async objects, preventing cross-event-loop failures
- **2026-02-10**: Proper Unicode emoji encoding in Telegram analysis messages
- **2026-02-10**: Security dependency updates: aiohttp 3.13.3, Flask 3.1.2, gunicorn 24.1.1, werkzeug 3.1.5
- **2026-02-10**: Production workflow configured with gunicorn WSGI server on port 5000

## External Dependencies
- **Kraken API**: For cryptocurrency exchange operations (KRAKEN_API_KEY, KRAKEN_API_SECRET).
- **Telegram Bot API**: For real-time notifications and interactive commands (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID).
- **alternative.me API**: Used for fetching the Fear & Greed Index.
- **Replit Auth**: For user authentication and session management.
- **PostgreSQL**: For user profile and application data storage.