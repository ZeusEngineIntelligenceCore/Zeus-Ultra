#!/usr/bin/env python3
"""
ZEUS AUTONOMOUS TRADING BOT
Main entry point for the trading bot
"""

import asyncio
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from src.zeus.core.bot import ZeusBot

app = Flask(__name__)

bot_status = {"running": False, "mode": "UNKNOWN"}


@app.route("/")
def health_check():
    return {"status": "healthy", "bot_running": bot_status["running"], "mode": bot_status["mode"]}, 200


@app.route("/health")
def health():
    return {"status": "ok"}, 200


def run_flask():
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, use_reloader=False)


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
    print("")

    bot_status["mode"] = mode
    bot_status["running"] = True

    bot = ZeusBot(
        kraken_key=kraken_key,
        kraken_secret=kraken_secret,
        telegram_token=telegram_token,
        telegram_chat_id=telegram_chat_id,
        mode=mode
    )

    await bot.run_forever()


if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print("[HEALTH] Health check server started on port 5000")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)
