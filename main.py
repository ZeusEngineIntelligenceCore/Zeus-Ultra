#!/usr/bin/env python3
"""
ZEUS AUTONOMOUS TRADING BOT
Main entry point for the trading bot
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.zeus.core.bot import ZeusBot


async def main():
    print("=" * 60)
    print("  ZEUS AUTONOMOUS TRADING BOT")
    print("  Advanced Cryptocurrency Trading System")
    print("=" * 60)

    kraken_key = os.environ.get("KRAKEN_API_KEY", "")
    kraken_secret = os.environ.get("KRAKEN_API_SECRET", "")
    telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    mode = os.environ.get("TRADING_MODE", "PAPER")

    if not kraken_key or not kraken_secret:
        print("\n[WARNING] Kraken API credentials not set.")
        print("The bot will run in READ-ONLY mode (market scanning only).")
        print("Set KRAKEN_API_KEY and KRAKEN_API_SECRET to enable trading.")
        print("")

    print(f"\n[MODE] Running in {mode} mode")
    print(f"[TELEGRAM] {'Enabled' if telegram_token else 'Disabled'}")
    print("")

    bot = ZeusBot(
        kraken_key=kraken_key,
        kraken_secret=kraken_secret,
        telegram_token=telegram_token,
        telegram_chat_id=telegram_chat_id,
        mode=mode
    )

    await bot.run_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)
