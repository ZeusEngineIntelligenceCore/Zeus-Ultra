# Zeus Autonomous Trading Bot

## Overview
Zeus is a professional-grade autonomous cryptocurrency trading bot designed for the Kraken exchange with multi-exchange architecture. It features advanced technical analysis, pre-breakout detection, and real-time Telegram alerts.

## Project Structure
```
├── main.py                    # Main entry point with Flask routes
├── app.py                     # Flask app configuration with database
├── models.py                  # Database models (User, OAuth)
├── replit_auth.py             # Replit Auth integration
├── templates/                 # HTML templates
│   ├── index.html            # Main landing/dashboard page
│   └── 403.html              # Error page
├── src/zeus/
│   ├── core/
│   │   ├── bot.py            # Main trading bot engine
│   │   └── state.py          # State management
│   ├── exchanges/
│   │   ├── base.py           # Abstract exchange interface
│   │   └── kraken.py         # Kraken API integration
│   ├── indicators/
│   │   ├── math_kernel.py    # 25+ technical indicators
│   │   └── prebreakout_detector.py  # CAT-9 pre-breakout detection
│   ├── strategies/
│   │   ├── signal_generator.py  # Trading signal generation
│   │   └── risk_manager.py   # Risk management & position sizing
│   └── alerts/
│       └── telegram_bot.py   # Telegram notifications
├── data/                      # Bot state storage
└── logs/                      # Log files
```

## Features
- **25+ Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, VWAP, Ichimoku, and more
- **Pre-Breakout Detection**: CAT-9 Ultra scoring system for identifying breakout candidates
- **Multi-Timeframe Analysis**: Confluence across 5m, 15m, 1h, 4h timeframes
- **Risk Management**: Kelly Criterion position sizing, max drawdown controls
- **Order Book Analysis**: Optimal entry/exit price detection
- **Telegram Alerts**: Real-time notifications for signals and trades
- **Paper/Live Modes**: Safe testing before live trading

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
- Enhanced signal formulas with multi-confluence scoring and advanced trend analysis (February 2026)
- Upgraded prebreakout detector with momentum boost and volume confirmation (February 2026)
- Enhanced continuous trading with 20 max positions (February 2026)
- Added holdings sync to track all owned tokens automatically (February 2026)
- Optimized for higher profit sells (3-4% minimum profit targets) (February 2026)
- Added Replit Auth integration (January 2026)
- Initial implementation (January 2026)

## Trading Configuration
- **Max Open Positions**: 20 trades
- **Minimum Profit Target**: 3-4% before trailing stop activates
- **Holdings Monitoring**: Automatically detects and tracks all owned tokens
- **Scan Interval**: 30 seconds continuous market scanning
- **Strategy**: Day trading with pre-breakout detection
- **Loss Prevention**: Bot will NOT sell at a loss - waits for recovery
- **Manual Order Protection**: Manually placed orders require 5%+ profit to auto-sell
- **AI/ML Learning**: Continuously learns from trade outcomes to improve decisions

## AI/ML Learning System
The bot includes a machine learning engine that:
- Tracks performance by symbol, strategy, and time of day
- Adjusts confidence thresholds based on historical win rates
- Identifies optimal trading hours from past performance
- Blocks trades on symbols with historically low win rates
- Runs learning cycles every 100 trading cycles to update insights
