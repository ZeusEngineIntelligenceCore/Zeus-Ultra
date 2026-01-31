# Zeus Autonomous Trading Bot

## Overview
Zeus is a professional-grade autonomous cryptocurrency trading bot designed for the Kraken exchange with multi-exchange architecture. It features advanced technical analysis, pre-breakout detection, and real-time Telegram alerts.

## Project Structure
```
├── main.py                    # Main entry point
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
- Initial implementation (January 2026)
