# Zeus Autonomous Trading Bot

## Overview
Zeus is a professional-grade autonomous cryptocurrency trading bot designed for the Kraken exchange with a multi-exchange architecture. Its core purpose is to identify and capitalize on pre-breakout trading opportunities using advanced technical analysis. Key capabilities include sophisticated market microstructure analysis, institutional-grade risk management, smart order execution, and AI/ML-driven learning. The project aims to provide a robust, self-improving trading solution for cryptocurrency markets, offering capabilities typically reserved for institutional traders to a broader audience.

## User Preferences
- The bot should run automatically via the configured workflow.
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
- **KPI Analytics System**: Tracks 10 critical performance metrics including Sharpe ratio, Sortino ratio, max drawdown, win rate, profit factor, expectancy, Calmar ratio, trade frequency, average hold times, and symbol-level performance. Records daily returns for accurate drawdown calculations.
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

## External Dependencies
- **Kraken API**: For cryptocurrency exchange operations (KRAKEN_API_KEY, KRAKEN_API_SECRET).
- **Telegram Bot API**: For real-time notifications and interactive commands (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID).
- **alternative.me API**: Used for fetching the Fear & Greed Index.
- **Replit Auth**: For user authentication and session management.
- **PostgreSQL**: For user profile and application data storage.