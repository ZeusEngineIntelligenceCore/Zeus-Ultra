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
- **Core Trading Engine**: Utilizes over 40 technical indicators and a 30 KPI pre-breakout scoring system with weighted multi-factor analysis. It focuses on the top 20 priority candidates and performs multi-timeframe analysis (5m, 15m, 1h, 4h, 1d) with 500 candles each.
- **Advanced Market Microstructure**: Incorporates Order Flow Imbalance (OFI), VPIN Toxicity Detection, Iceberg Order Detection, Market Maker Bias Analysis, and Liquidity Scoring.
- **Institutional-Grade Risk Management**: Features Dynamic Kelly Criterion, Value at Risk (VaR), Expected Shortfall (ES), Correlation Matrix, Circuit Breakers, and Monte Carlo Simulations.
- **Smart Order Execution**: Implements TWAP, VWAP, and Iceberg order algorithms, along with ML-based slippage prediction and adaptive execution strategies.
- **Market Regime Detection**: Uses Hidden Markov Models (HMM) and GARCH for volatility clustering, enabling automatic strategy adaptation based on trend vs. range and momentum vs. mean-reversion classifications.
- **Advanced Mathematical Analysis**: Applies Hurst Exponent, Fractal Support/Resistance, Wavelet Denoising, Shannon Entropy, and Information Ratio for signal quality assessment.
- **AI/ML Learning System**: Features dual learning engines for the trading bot and Telegram. These engines track performance by symbol, strategy, and time of day, adapt confidence thresholds, detect anomalies, and continuously improve decision-making based on trade outcomes and user interactions.
- **Trading Configuration**: Supports a maximum of 100 open positions, a minimum profit target of 3-4%, continuous holdings monitoring, full market scans every 300 seconds, and position scans every 60 seconds.
- **Authentication**: Uses Replit Auth for user authentication, supporting various social logins and email/password, with protected routes.
- **Production Server**: Employs a Gunicorn WSGI production server.
- **Timezone**: All timestamps are configured for Los Angeles timezone (America/Los_Angeles).

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