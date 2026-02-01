#!/usr/bin/env python3
"""
ZEUS AUTONOMOUS TRADING BOT
Main entry point - Production WSGI server with background trading bot
"""

import asyncio
import os
import sys
import threading
import json
from datetime import datetime
import pytz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import render_template, jsonify, request
from flask_login import current_user
from app import app
from replit_auth import make_replit_blueprint, require_login
from src.zeus.core.bot import ZeusBot

app.register_blueprint(make_replit_blueprint(), url_prefix="/auth")

LA_TZ = pytz.timezone('America/Los_Angeles')

bot_instance = None
bot_thread = None
bot_lock = threading.Lock()

bot_status = {
    "running": False,
    "mode": "UNKNOWN",
    "error": None,
    "balance": 0,
    "holdings_count": 0,
    "active_trades": 0,
    "total_profit": 0,
    "candidates_count": 0,
    "last_scan": None,
    "uptime_start": None
}


def get_la_time():
    return datetime.now(LA_TZ)


def format_la_time(dt=None):
    if dt is None:
        dt = get_la_time()
    return dt.strftime("%Y-%m-%d %I:%M:%S %p PT")


@app.route("/")
def index():
    return render_template("index.html", user=current_user, bot_status=bot_status, current_time=format_la_time())


@app.route("/health")
def health():
    return {"status": "ok", "time": format_la_time()}, 200


@app.route("/status")
def status():
    with bot_lock:
        status_copy = bot_status.copy()
        status_copy["current_time"] = format_la_time()
        if bot_instance and bot_instance.running:
            try:
                state = bot_instance.get_status()
                status_copy["balance"] = state.get("balance", 0)
                status_copy["active_trades"] = len(state.get("active_trades", {}))
                status_copy["candidates_count"] = len(state.get("candidates", []))
                status_copy["total_profit"] = state.get("total_profit", 0)
                status_copy["cycle_count"] = state.get("cycle_count", 0)
            except Exception:
                pass
    return jsonify(status_copy), 200


@app.route("/api/trades")
@require_login
def api_trades():
    if not bot_instance:
        return jsonify({"trades": [], "error": "Bot not running"}), 200
    try:
        state = bot_instance.get_status()
        trades = []
        for trade_id, trade in state.get("active_trades", {}).items():
            trades.append({
                "id": trade_id,
                "symbol": trade.get("symbol"),
                "side": trade.get("side"),
                "entry_price": trade.get("entry_price"),
                "size": trade.get("size"),
                "stop_loss": trade.get("stop_loss"),
                "take_profit": trade.get("take_profit"),
                "entry_time": trade.get("entry_time"),
                "status": trade.get("status")
            })
        return jsonify({"trades": trades, "count": len(trades)}), 200
    except Exception as e:
        return jsonify({"trades": [], "error": str(e)}), 200


@app.route("/api/candidates")
@require_login
def api_candidates():
    if not bot_instance:
        return jsonify({"candidates": [], "error": "Bot not running"}), 200
    try:
        state = bot_instance.get_status()
        candidates = state.get("candidates", [])[:20]
        return jsonify({"candidates": candidates, "count": len(candidates)}), 200
    except Exception as e:
        return jsonify({"candidates": [], "error": str(e)}), 200


@app.route("/api/bot/toggle", methods=["POST"])
@require_login
def toggle_bot():
    global bot_instance, bot_thread
    with bot_lock:
        if bot_status["running"]:
            if bot_instance:
                bot_instance.running = False
            bot_status["running"] = False
            bot_status["uptime_start"] = None
            return jsonify({"status": "stopped", "message": "Bot stopped"}), 200
        else:
            bot_thread = threading.Thread(target=run_trading_bot, daemon=True)
            bot_thread.start()
            return jsonify({"status": "starting", "message": "Bot starting..."}), 200


@app.route("/dashboard")
@require_login
def dashboard():
    return render_template("dashboard.html", user=current_user, bot_status=bot_status, current_time=format_la_time())


def run_trading_bot():
    global bot_instance
    
    async def start_bot():
        global bot_instance
        print("=" * 60)
        print("  ZEUS AUTONOMOUS TRADING BOT")
        print("  Advanced Cryptocurrency Trading System")
        print(f"  Started: {format_la_time()}")
        print("=" * 60)

        kraken_key = os.environ.get("KRAKEN_API_KEY", "")
        kraken_secret = os.environ.get("KRAKEN_API_SECRET", "")
        telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        mode = os.environ.get("TRADING_MODE", "PAPER")

        if not kraken_key or not kraken_secret:
            print("\n[WARNING] Kraken API credentials not set.")
            print("The bot will run in READ-ONLY mode (market scanning only).")

        print(f"\n[MODE] Running in {mode} mode")

        with bot_lock:
            bot_status["mode"] = mode
            bot_status["running"] = True
            bot_status["uptime_start"] = format_la_time()
            bot_status["error"] = None

        try:
            bot_instance = ZeusBot(
                kraken_key=kraken_key,
                kraken_secret=kraken_secret,
                telegram_token=telegram_token,
                telegram_chat_id=telegram_chat_id,
                mode=mode
            )
            await bot_instance.run_forever()
        except Exception as e:
            with bot_lock:
                bot_status["error"] = str(e)
                bot_status["running"] = False
            print(f"Bot error: {e}")

    asyncio.run(start_bot())


def run_server():
    port = int(os.environ.get("PORT", 5000))
    print(f"[SERVER] Starting production server on port {port}")
    
    from gunicorn.app.base import BaseApplication

    class GunicornApp(BaseApplication):
        def __init__(self, application, options=None):
            self.options = options or {}
            self.application = application
            super().__init__()

        def load_config(self):
            for key, value in self.options.items():
                if key in self.cfg.settings and value is not None:
                    self.cfg.set(key.lower(), value)

        def load(self):
            return self.application

    options = {
        'bind': f'0.0.0.0:{port}',
        'workers': 2,
        'threads': 4,
        'timeout': 120,
        'accesslog': '-',
        'errorlog': '-',
        'loglevel': 'info',
        'preload_app': True
    }
    
    GunicornApp(app, options).run()


if __name__ == "__main__":
    bot_thread = threading.Thread(target=run_trading_bot, daemon=True)
    bot_thread.start()
    print("[BOT] Trading bot started in background thread")
    
    run_server()
