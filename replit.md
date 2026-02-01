# Zeus Autonomous Trading Bot

## Overview
Zeus is a professional-grade autonomous cryptocurrency trading bot designed for the Kraken exchange with multi-exchange architecture. It features advanced technical analysis, pre-breakout detection, and real-time Telegram alerts.

## Project Structure
```
├── main.py                    # Main entry point with Gunicorn production server
├── app.py                     # Flask app configuration with database
├── models.py                  # Database models (User, OAuth)
├── replit_auth.py             # Replit Auth integration
├── templates/                 # HTML templates
│   ├── index.html            # Main landing page
│   ├── dashboard.html        # Interactive trading dashboard with toggle
│   └── 403.html              # Error page
├── src/zeus/
│   ├── core/
│   │   ├── bot.py            # Main trading bot engine
│   │   └── state.py          # State management
│   ├── exchanges/
│   │   ├── base.py           # Abstract exchange interface
│   │   └── kraken.py         # Kraken API integration
│   ├── indicators/
│   │   ├── math_kernel.py    # 40+ technical indicators
│   │   ├── prebreakout_detector.py  # 23-KPI pre-breakout detection
│   │   ├── microstructure.py # Order flow & VPIN analysis
│   │   └── advanced_math.py  # Hurst, fractals, wavelets
│   ├── strategies/
│   │   ├── signal_generator.py  # Trading signal generation
│   │   └── risk_manager.py   # Risk management & position sizing
│   ├── risk/
│   │   └── advanced_risk.py  # VaR, Kelly, correlation matrix
│   ├── execution/
│   │   └── smart_execution.py # TWAP/VWAP algorithms
│   ├── ml/
│   │   ├── learning_engine.py # Trade outcome learning
│   │   └── regime_detector.py # HMM/GARCH regime detection
│   ├── data/
│   │   └── alternative_data.py # Fear&Greed, funding rates
│   └── alerts/
│       └── telegram_bot.py   # Telegram notifications
├── data/                      # Bot state storage
└── logs/                      # Log files
```

## Features

### Core Trading Engine
- **40+ Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, VWAP, Ichimoku, Supertrend, Aroon, Vortex, Elder Ray, and many more
- **30 KPI Pre-Breakout Scoring**: Comprehensive analysis with weighted multi-factor scoring (Williams %R, Stoch RSI, MFI, OBV Trend, Pivot Distance, Fibonacci Levels, Parabolic SAR + 23 original KPIs)
- **Top-20 Priority Focus**: Identifies and tracks the best 20 candidates until exhausted
- **Multi-Timeframe Analysis**: Confluence across 5m, 15m, 1h, 4h, 1d timeframes with 500 candles each

### Advanced Market Microstructure (NEW)
- **Order Flow Imbalance (OFI)**: Detects buying/selling pressure from order book changes
- **VPIN Toxicity Detection**: Volume-Synchronized Probability of Informed Trading
- **Iceberg Order Detection**: Identifies hidden institutional orders
- **Market Maker Bias Analysis**: Tracks smart money positioning
- **Liquidity Scoring**: Real-time liquidity assessment for optimal execution

### Institutional-Grade Risk Management (NEW)
- **Dynamic Kelly Criterion**: Volatility-adjusted position sizing
- **Value at Risk (VaR)**: 95% and 99% confidence intervals
- **Expected Shortfall (ES)**: Tail risk measurement
- **Correlation Matrix**: Portfolio-wide risk assessment
- **Circuit Breakers**: Automatic trading halt on max drawdown or daily loss limits
- **Monte Carlo Simulations**: Drawdown probability forecasting

### Smart Order Execution (NEW)
- **TWAP Algorithm**: Time-Weighted Average Price execution
- **VWAP Algorithm**: Volume-Weighted Average Price execution
- **Iceberg Orders**: Hidden order execution for large positions
- **Slippage Prediction**: ML-based slippage estimation and minimization
- **Adaptive Execution**: Strategy selection based on market conditions

### Market Regime Detection (NEW)
- **Hidden Markov Models (HMM)**: State transition detection
- **GARCH Volatility Clustering**: Volatility regime identification
- **Trend vs Range Classification**: Automatic strategy adaptation
- **Momentum vs Mean-Reversion**: Dynamic mode switching

### Advanced Mathematical Analysis (NEW)
- **Hurst Exponent**: Trend persistence measurement (0-1 scale)
- **Fractal Support/Resistance**: Automatic S/R level detection
- **Wavelet Denoising**: Noise filtering for cleaner signals
- **Shannon Entropy**: Market efficiency measurement
- **Information Ratio**: Signal quality assessment

### Alternative Data Integration (NEW)
- **Fear & Greed Index**: Market sentiment from alternative.me API
- **Funding Rate Analysis**: Perpetual futures positioning signals
- **Open Interest Tracking**: Derivatives market analysis
- **Whale Activity Monitoring**: Large transaction tracking

### AI/ML Learning System
- **Dual Learning Engines**: Both trading bot and Telegram learn from your trades
- **Performance Tracking**: By symbol, strategy, and time of day
- **Adaptive Confidence**: Thresholds adjust based on historical win rates
- **Anomaly Detection**: Filters unusual market conditions

### Additional Features
- **Telegram Alerts with Learning**: Adapts to your trading preferences and patterns
- **Paper/Live Modes**: Safe testing before live trading
- **Production Gunicorn Server**: Single worker to prevent duplicate bot instances

## Required Secrets
- `KRAKEN_API_KEY`: Your Kraken API key
- `KRAKEN_API_SECRET`: Your Kraken API secret
- `TELEGRAM_BOT_TOKEN`: Telegram bot token (optional)
- `TELEGRAM_CHAT_ID`: Your Telegram chat ID (optional)
- `TRADING_MODE`: Set to "PAPER" or "LIVE"
- `SESSION_SECRET`: Session encryption key (auto-generated)

## Authentication
The app uses Replit Auth for user authentication, supporting:
- Google, GitHub, X (Twitter), Apple login
- Email/password authentication
- Protected routes with `@require_login` decorator
- User profile storage in PostgreSQL database

## Running the Bot
The bot runs automatically via the configured workflow. It will:
1. Connect to Kraken API
2. Scan all USD trading pairs
3. Detect pre-breakout conditions
4. Generate trading signals
5. Execute trades (if in LIVE mode)
6. Send Telegram alerts

## Risk Warning
- Cryptocurrency trading involves significant risk
- Past performance does not guarantee future results
- Always test with PAPER mode first
- Never trade more than you can afford to lose

## Recent Changes
- TELEGRAM STABILITY FIX: Fixed StateManager attribute errors - /trades and /report commands now correctly access active_trades (February 2026)
- Fixed Telegram polling conflict by properly deleting webhook before starting polling (February 2026)
- Added get_daily_insights() method to TelegramLearningEngine for performance analytics (February 2026)
- Fixed cmd_report to use correct send_daily_summary parameters (February 2026)
- MAJOR TELEGRAM UPGRADE: Added interactive commands (/status, /portfolio, /trades, /candidates, /performance, /settings, /help, /report) with inline keyboard buttons (February 2026)
- Added Telegram command listener with callback query handlers for quick action buttons (February 2026)
- Added scheduled daily/weekly performance reports at 8 PM PT (February 2026)
- MAJOR ML UPGRADE: Added advanced feature engineering, pattern recognition, and predictive profit/loss models (February 2026)
- Added ML predict_trade_outcome() with win probability estimation and expected PnL calculation (February 2026)
- Added pattern cluster detection to identify winning vs losing trade characteristics (February 2026)
- Added adaptive confidence thresholds based on recent performance and market conditions (February 2026)
- Added comprehensive position scoring combining all ML factors (February 2026)
- Added Telegram error alert rate limiting: Same error only sent once per 5 minutes to prevent spam (February 2026)
- Fixed holdings persistence: Holdings now saved to and loaded from state file correctly (February 2026)
- Fixed holdings sync: Only creates positions when sell orders are successfully placed (February 2026)
- Added open orders sync: Fetches existing Kraken orders before placing new ones to avoid duplicate orders (February 2026)
- Added ghost position detection: Removes positions with no tokens or failed sell order attempts (February 2026)
- Added robust error handling for locked/staked tokens that can't be sold (February 2026)
- Expanded to 30 KPIs for pre-breakout detection (+7 new: Williams %R, Stoch RSI, MFI, OBV Trend, Pivot Distance, Fibonacci Levels, Parabolic SAR) (February 2026)
- Manual order protection: Bot now NEVER auto-closes manually placed orders (February 2026)
- FULL INTEGRATION: All 6 institutional modules now wired into runtime (February 2026)
  - Advanced Risk Engine: Kelly sizing + circuit breakers in execute_trade/scan_markets
  - Smart Execution: Strategy selection (TWAP/VWAP/market) based on liquidity/volatility
  - Microstructure: Order flow imbalance (OFI) integrated into signal scoring
  - Alternative Data: Fear & Greed fetched every 30min, affects circuit breaker
  - Regime Detector: HMM/GARCH analysis influences signal scoring
  - Advanced Math: Hurst exponent + wavelet denoising in signal analysis
- Added production Gunicorn WSGI server (February 2026)
- Implemented Los Angeles timezone (America/Los_Angeles) for all timestamps (February 2026)
- Added interactive dashboard with bot toggle on/off switch (February 2026)
- Immediate sell order placement when buy orders fill (February 2026)
- Real-time dashboard with active trades and top candidates panels (February 2026)
- Sell order monitoring and dynamic adjustment based on market conditions (February 2026)
- Expanded to 40+ technical indicators in MathKernel (February 2026)
- Enhanced PreBreakout detector with 23 KPIs for precision scoring (February 2026)
- Added Top-20 priority candidate focus system (February 2026)
- Added Telegram Learning Engine for adaptive user preferences (February 2026)
- Dual learning system: both bot and Telegram learn from trades (February 2026)
- Fixed limit order fill verification with 10-second polling and market fallback (February 2026)
- Increased candle depth to 500 across all timeframes for deeper market analysis (February 2026)
- Added enforce_trade_only_alerts() to guarantee only trade open/close notifications (February 2026)
- Added Telegram alert batching - non-urgent alerts held for 30 min summaries (February 2026)
- Enhanced signal formulas with multi-confluence scoring and advanced trend analysis (February 2026)
- Upgraded prebreakout detector with momentum boost and volume confirmation (February 2026)
- Enhanced continuous trading with 100 max positions (February 2026)
- Added holdings sync to track all owned tokens automatically (February 2026)
- Optimized for higher profit sells (3-4% minimum profit targets) (February 2026)
- Added Replit Auth integration (January 2026)
- Initial implementation (January 2026)

## Trading Configuration
- **Max Open Positions**: 100 trades
- **Minimum Profit Target**: 3-4% before trailing stop activates
- **Holdings Monitoring**: Automatically detects and tracks all owned tokens
- **Full Market Scan**: Every 300 seconds (5 minutes) with all timeframes
- **Position Scan**: Every 60 seconds for active trades with multi-timeframe analysis
- **Timeframes**: 5m, 15m, 1h, 4h, 1d - all analyzed for each candidate
- **Rate Limiter**: 15 public calls/sec, 8 private calls/sec
- **Strategy**: Day trading with pre-breakout detection
- **Loss Prevention**: Bot will NOT sell at a loss - waits for recovery
- **Manual Order Protection**: Bot will NEVER auto-close manually placed orders - full user control preserved
- **AI/ML Learning**: Continuously learns from trade outcomes to improve decisions

## AI/ML Learning System
The bot includes dual machine learning engines:

### Trading Bot Learning Engine
- Tracks performance by symbol, strategy, and time of day
- Adjusts confidence thresholds based on historical win rates
- Identifies optimal trading hours from past performance
- Blocks trades on symbols with historically low win rates
- Runs learning cycles every 100 trading cycles to update insights

### Telegram Learning Engine
- Learns preferred symbols from successful trades
- Identifies and avoids consistently unprofitable symbols
- Tracks optimal trading hours based on your profits
- Adjusts profit thresholds from your winning patterns
- Learns optimal hold durations for maximum profit
- Identifies your best performing strategies
- Both engines sync every 100 cycles for continuous improvement
