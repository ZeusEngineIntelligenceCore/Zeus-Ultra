#!/usr/bin/env python3
"""
ZEUS AUTONOMOUS TRADING BOT
Main entry point - Flask web server with background trading bot
"""

import asyncio
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from src.zeus.core.bot import ZeusBot

app = Flask(__name__)

bot_status = {"running": False, "mode": "UNKNOWN", "error": None}


@app.route("/")
def health_check():
    return {"status": "healthy", "bot_running": bot_status["running"], "mode": bot_status["mode"]}, 200


@app.route("/health")
def health():
    return {"status": "ok"}, 200


@app.route("/status")
def status():
    return bot_status, 200


@app.route("/api")
def api():
    return {"status": "ok"}, 200


def run_trading_bot():
    async def start_bot():
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

        try:
            bot = ZeusBot(
                kraken_key=kraken_key,
                kraken_secret=kraken_secret,
                telegram_token=telegram_token,
                telegram_chat_id=telegram_chat_id,
                mode=mode
            )
            await bot.run_forever()
        except Exception as e:
            bot_status["error"] = str(e)
            bot_status["running"] = False
            print(f"Bot error: {e}")

    asyncio.run(start_bot())


if __name__ == "__main__":
    bot_thread = threading.Thread(target=run_trading_bot, daemon=True)
    bot_thread.start()
    print("[BOT] Trading bot started in background thread")
    
    port = int(os.environ.get("PORT", 5000))
    print(f"[SERVER] Starting web server on port {port}")
    app.run(host="0.0.0.0", port=port, use_reloader=False)
