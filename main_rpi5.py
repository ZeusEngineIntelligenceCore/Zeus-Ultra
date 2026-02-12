#!/usr/bin/env python3
"""
ZEUS AUTONOMOUS TRADING BOT - Raspberry Pi 5 Version
Main entry point - Standalone version without Replit dependencies
Run from project root: python main_rpi5.py
"""

import asyncio
import os
import sys
import threading
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

try:
    import pytz
    LA_TZ = pytz.timezone('America/Los_Angeles')
except ImportError:
    LA_TZ = None
    print("[WARNING] pytz not installed, using UTC")

from flask import Flask, render_template, jsonify, request

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24).hex())

# Import bot after Flask setup
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
    if LA_TZ:
        return datetime.now(LA_TZ)
    return datetime.utcnow()


def format_la_time(dt=None):
    if dt is None:
        dt = get_la_time()
    return dt.strftime("%Y-%m-%d %I:%M:%S %p PT")


@app.route("/")
def index():
    try:
        return render_template("index.html",
                               user=None,
                               bot_status=bot_status,
                               current_time=format_la_time())
    except Exception as e:
        return f"""
        <html><head><title>Zeus Trading Bot</title></head>
        <body style="font-family: sans-serif; padding: 20px;">
        <h1>Zeus Trading Bot</h1>
        <p>Status: {'Running' if bot_status['running'] else 'Stopped'}</p>
        <p>Mode: {bot_status['mode']}</p>
        <p>Balance: ${bot_status['balance']:.2f}</p>
        <p>Active Trades: {bot_status['active_trades']}</p>
        <p><a href="/status">JSON Status</a> | <a href="/dashboard">Dashboard</a></p>
        </body></html>
        """


@app.route("/health")
def health():
    return {"status": "ok", "time": format_la_time()}, 200


@app.route("/status")
def status():
    with bot_lock:
        status_copy = bot_status.copy()
        status_copy["current_time"] = format_la_time()

        state_loaded = False
        if bot_instance and bot_instance.running:
            try:
                state = bot_instance.get_status()
                stats = state.get("stats", {})
                holdings = state.get("holdings", {})
                status_copy["balance"] = stats.get("equity", 0)
                status_copy["holdings_count"] = len([
                    k for k, v in holdings.items()
                    if k not in ("USD", "ZUSD", "USDT", "USDC") and v > 0.0001
                ])
                status_copy["active_trades"] = len(
                    state.get("active_trades", {}))
                status_copy["candidates_count"] = len(
                    state.get("candidates", []))
                status_copy["total_profit"] = stats.get("total_pnl", 0)
                status_copy["cycle_count"] = state.get("cycle_count", 0)
                status_copy["wins"] = stats.get("wins", 0)
                status_copy["losses"] = stats.get("losses", 0)
                status_copy["mode"] = stats.get(
                    "mode", status_copy.get("mode", "UNKNOWN"))
                if hasattr(bot_instance, 'alt_data') and bot_instance.alt_data:
                    fg = bot_instance.alt_data.get_fear_greed()
                    if fg:
                        status_copy["fear_greed"] = {
                            "value": fg.value,
                            "classification": fg.classification
                        }
                state_loaded = True
            except Exception:
                pass

        if not state_loaded:
            try:
                state_file = Path("data/bot_state.json")
                if state_file.exists():
                    with open(state_file, 'r') as f:
                        data = json.load(f)
                    status_copy["balance"] = round(data.get("equity", 0), 2)
                    holdings = data.get("holdings", {})
                    status_copy["holdings_count"] = len([
                        k for k, v in holdings.items()
                        if k not in ("USD", "ZUSD", "USDT",
                                     "USDC") and v > 0.0001
                    ])
                    status_copy["active_trades"] = len(
                        data.get("active_trades", {}))
                    status_copy["total_profit"] = round(
                        data.get("total_pnl", 0), 2)
                    status_copy["wins"] = data.get("wins", 0)
                    status_copy["losses"] = data.get("losses", 0)
                    config = data.get("config", {})
                    status_copy["mode"] = config.get(
                        "mode", status_copy.get("mode", "UNKNOWN"))
            except Exception:
                pass

    return jsonify(status_copy), 200


@app.route("/api/trades")
def api_trades():
    trades = []
    try:
        if bot_instance:
            state = bot_instance.get_status()
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
                    "status": trade.get("status"),
                    "is_manual": trade.get("is_manual", False),
                    "protected": trade.get("protected", False)
                })
        if not trades:
            state_file = Path("data/bot_state.json")
            if state_file.exists():
                with open(state_file, 'r') as f:
                    data = json.load(f)
                for trade_id, trade in data.get("active_trades", {}).items():
                    trades.append({
                        "id": trade_id,
                        "symbol": trade.get("symbol"),
                        "side": trade.get("side"),
                        "entry_price": trade.get("entry_price"),
                        "size": trade.get("size"),
                        "stop_loss": trade.get("stop_loss"),
                        "take_profit": trade.get("take_profit"),
                        "entry_time": trade.get("entry_time"),
                        "status": trade.get("status"),
                        "is_manual": trade.get("is_manual", False),
                        "protected": trade.get("protected", False)
                    })
        return jsonify({"trades": trades, "count": len(trades)}), 200
    except Exception as e:
        return jsonify({"trades": [], "error": str(e)}), 200


@app.route("/api/holdings")
def api_holdings():
    holdings_data = {}
    active_trades = {}
    try:
        if bot_instance:
            state = bot_instance.get_status()
            holdings_data = state.get("holdings", {})
            active_trades = state.get("active_trades", {})
        if not holdings_data:
            state_file = Path("data/bot_state.json")
            if state_file.exists():
                with open(state_file, 'r') as f:
                    data = json.load(f)
                holdings_data = data.get("holdings", {})
                active_trades = data.get("active_trades", {})
        price_cache = {}
        if bot_instance and hasattr(bot_instance,
                                    'exchange') and bot_instance.exchange:
            price_cache = getattr(bot_instance.exchange, '_ticker_cache', {})
        holdings = []

        skip_symbols = {"USD", "ZUSD", "EUR", "ZEUR", "USDT", "USDC", "DAI"}

        for symbol, amount in holdings_data.items():
            if symbol in skip_symbols or amount < 0.0001:
                continue

            clean_symbol = symbol
            for prefix in ("X", "Z"):
                if clean_symbol.startswith(prefix) and len(clean_symbol) > 3:
                    clean_symbol = clean_symbol[1:]
                    break

            price = 0.0
            value_usd = 0.0

            for pair_suffix in ["USD", "USDT", "USDC"]:
                cache_key = f"{symbol}{pair_suffix}"
                if cache_key in price_cache:
                    cached = price_cache[cache_key]
                    if hasattr(cached, 'last'):
                        price = cached.last
                    elif isinstance(cached, dict):
                        price = cached.get('last', cached.get('price', 0))
                    if price > 0:
                        break

            if price == 0:
                for trade_id, trade in active_trades.items():
                    trade_symbol = trade.get("symbol", "")
                    if trade_symbol.startswith(
                            symbol) or trade_symbol.startswith(clean_symbol):
                        price = trade.get("entry_price", 0)
                        if price > 0:
                            break

            value_usd = amount * price if price > 0 else 0

            holdings.append({
                "symbol": symbol,
                "amount": amount,
                "price": price,
                "value_usd": value_usd,
                "pnl_pct": 0
            })

        holdings.sort(key=lambda x: x.get("value_usd", 0), reverse=True)
        return jsonify({
            "holdings": holdings[:20],
            "count": len(holdings)
        }), 200
    except Exception as e:
        return jsonify({"holdings": [], "error": str(e)}), 200


recent_logs = []


@app.route("/api/logs")
def api_logs():
    global recent_logs
    try:
        log_file = Path("logs/zeus.log")
        if log_file.exists():
            with open(log_file, 'r') as f:
                lines = f.readlines()[-50:]
                recent_logs = [l.strip()[:150] for l in lines[-30:]]
    except Exception:
        pass
    return jsonify({"logs": recent_logs}), 200


@app.route("/api/candidates")
def api_candidates():
    if not bot_instance:
        return jsonify({"candidates": [], "error": "Bot not running"}), 200
    try:
        state = bot_instance.get_status()
        candidates = state.get("candidates", [])[:20]
        return jsonify({
            "candidates": candidates,
            "count": len(candidates)
        }), 200
    except Exception as e:
        return jsonify({"candidates": [], "error": str(e)}), 200


@app.route("/api/bot/toggle", methods=["POST"])
def toggle_bot():
    global bot_instance, bot_thread
    with bot_lock:
        if bot_status["running"]:
            if bot_instance:
                bot_instance.running = False
            bot_status["running"] = False
            bot_status["uptime_start"] = None
            return jsonify({
                "status": "stopped",
                "message": "Bot stopped"
            }), 200
        else:
            bot_thread = threading.Thread(target=run_trading_bot, daemon=True)
            bot_thread.start()
            return jsonify({
                "status": "starting",
                "message": "Bot starting..."
            }), 200


@app.route("/dashboard")
def dashboard():
    try:
        return render_template("dashboard.html",
                               user=None,
                               bot_status=bot_status,
                               current_time=format_la_time())
    except Exception:
        return f"""
        <html><head><title>Zeus Dashboard</title>
        <style>
            body {{ font-family: sans-serif; padding: 20px; background: #1a1a2e; color: #eee; }}
            .card {{ background: #16213e; padding: 15px; margin: 10px 0; border-radius: 8px; }}
            h1 {{ color: #e94560; }}
            .stat {{ display: inline-block; margin: 10px 20px; }}
            .stat-value {{ font-size: 24px; font-weight: bold; color: #0f3460; }}
            .green {{ color: #4ecca3; }}
            .red {{ color: #e94560; }}
        </style>
        </head>
        <body>
        <h1>Zeus Trading Bot Dashboard</h1>
        <div class="card">
            <div class="stat"><span class="stat-value {'green' if bot_status['running'] else 'red'}">{'Running' if bot_status['running'] else 'Stopped'}</span><br>Status</div>
            <div class="stat"><span class="stat-value">{bot_status['mode']}</span><br>Mode</div>
            <div class="stat"><span class="stat-value">${bot_status['balance']:.2f}</span><br>Balance</div>
            <div class="stat"><span class="stat-value">{bot_status['active_trades']}</span><br>Active Trades</div>
            <div class="stat"><span class="stat-value">{bot_status['holdings_count']}</span><br>Holdings</div>
        </div>
        <div class="card">
            <h3>API Endpoints</h3>
            <p><a href="/status" style="color:#4ecca3">/status</a> - Bot status JSON</p>
            <p><a href="/api/trades" style="color:#4ecca3">/api/trades</a> - Active trades</p>
            <p><a href="/api/holdings" style="color:#4ecca3">/api/holdings</a> - Current holdings</p>
            <p><a href="/api/candidates" style="color:#4ecca3">/api/candidates</a> - Trading candidates</p>
            <p><a href="/api/logs" style="color:#4ecca3">/api/logs</a> - Recent logs</p>
        </div>
        <p style="color:#888">Last updated: {format_la_time()}</p>
        </body></html>
        """


@app.route("/api/analyze/<symbol>")
def api_analyze_coin(symbol):
    if not bot_instance:
        return jsonify(
            {"error": "Bot not running - please start the bot first"}), 503

    symbol = symbol.upper().strip()
    if not symbol.endswith("USD"):
        symbol = f"{symbol}USD"

    async def find_valid_pair(exchange, base_symbol):
        markets = await exchange.fetch_markets()
        if not markets:
            return base_symbol

        base = base_symbol.replace("USD", "")
        possible_names = [
            base_symbol, f"X{base}ZUSD", f"X{base}USD", f"{base}ZUSD"
        ]

        for name in possible_names:
            if name in markets:
                return name

        for pair_name in markets.keys():
            if pair_name.upper().endswith("USD"):
                pair_base = pair_name.replace("USD",
                                              "").replace("ZUSD",
                                                          "").lstrip("X")
                if pair_base.upper() == base.upper():
                    return pair_name

        return base_symbol

    async def run_analysis():
        nonlocal symbol
        try:
            exchange = bot_instance.exchange
            prebreakout = bot_instance.prebreakout
            math_kernel = prebreakout.math

            valid_symbol = await find_valid_pair(exchange, symbol)
            symbol = valid_symbol

            timeframes = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
            all_analysis = {}

            for tf_name, tf_minutes in timeframes.items():
                try:
                    ohlcv = await exchange.fetch_ohlcv(symbol,
                                                       tf_name,
                                                       limit=500)
                    if not ohlcv or len(ohlcv) < 50:
                        continue

                    high = [c.high for c in ohlcv]
                    low = [c.low for c in ohlcv]
                    close = [c.close for c in ohlcv]
                    volume = [c.volume for c in ohlcv]

                    analysis = await prebreakout.analyze(
                        symbol, high, low, close, volume)

                    rsi = math_kernel.rsi(close)
                    macd_line, signal_line, macd_hist = math_kernel.macd(close)
                    bb_upper, bb_mid, bb_lower = math_kernel.bollinger_bands(
                        close)
                    atr = math_kernel.atr(high, low, close)
                    stoch_k, stoch_d = math_kernel.stochastic_rsi(close)
                    ema9 = math_kernel.ema(close, 9)
                    ema20 = math_kernel.ema(close, 20)
                    ema50 = math_kernel.ema(close, 50)
                    sma200 = math_kernel.sma(close, 200)
                    adx_val, plus_di, minus_di = math_kernel.adx(
                        high, low, close)

                    analysis["technical_indicators"] = {
                        "rsi":
                        round(rsi, 2),
                        "macd":
                        round(macd_line, 8),
                        "macd_signal":
                        round(signal_line, 8),
                        "macd_histogram":
                        round(macd_hist, 8),
                        "bb_upper":
                        round(bb_upper, 8),
                        "bb_middle":
                        round(bb_mid, 8),
                        "bb_lower":
                        round(bb_lower, 8),
                        "bb_position":
                        round((close[-1] - bb_lower) / (bb_upper - bb_lower) *
                              100, 2) if (bb_upper - bb_lower) > 0 else 50,
                        "atr":
                        round(atr, 8),
                        "atr_pct":
                        round(atr / close[-1] *
                              100, 2) if close[-1] > 0 else 0,
                        "stoch_k":
                        round(stoch_k, 2),
                        "stoch_d":
                        round(stoch_d, 2),
                        "ema_9":
                        round(ema9, 8),
                        "ema_20":
                        round(ema20, 8),
                        "ema_50":
                        round(ema50, 8),
                        "sma_200":
                        round(sma200, 8),
                        "adx":
                        round(adx_val, 2),
                        "plus_di":
                        round(plus_di, 2),
                        "minus_di":
                        round(minus_di, 2)
                    }

                    price_change_24h = (
                        (close[-1] - close[-min(24, len(close))]) /
                        close[-min(24, len(close))]) * 100 if len(
                            close) > 24 else 0
                    high_24h = max(high[-min(24, len(high)):])
                    low_24h = min(low[-min(24, len(low)):])
                    avg_volume = sum(volume[-20:]) / 20 if len(
                        volume) >= 20 else sum(volume) / len(volume)
                    current_volume = volume[-1] if volume else 0

                    analysis["market_data"] = {
                        "current_price":
                        round(close[-1], 8),
                        "price_change_24h":
                        round(price_change_24h, 2),
                        "high_24h":
                        round(high_24h, 8),
                        "low_24h":
                        round(low_24h, 8),
                        "current_volume":
                        round(current_volume, 2),
                        "avg_volume":
                        round(avg_volume, 2),
                        "volume_ratio":
                        round(current_volume /
                              avg_volume, 2) if avg_volume > 0 else 0
                    }

                    all_analysis[tf_name] = analysis
                except Exception as e:
                    all_analysis[tf_name] = {"error": str(e)}

            if not all_analysis:
                return {"error": f"Could not fetch data for {symbol}"}

            valid_analyses = {
                k: v
                for k, v in all_analysis.items()
                if "error" not in v and v.get("current_price", 0) > 0
            }

            if not valid_analyses:
                return {"error": f"Failed to fetch data for {symbol}"}

            primary_tf = valid_analyses.get("15m") or valid_analyses.get(
                "1h") or list(valid_analyses.values())[0]

            return {
                "symbol": symbol,
                "timestamp": format_la_time(),
                "primary_analysis": primary_tf,
                "timeframes": valid_analyses,
                "success": True
            }
        except Exception as e:
            return {"error": str(e)}

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(run_analysis())
        loop.close()
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/pairs")
def api_pairs():
    if not bot_instance:
        return jsonify({"pairs": []}), 200
    try:
        pairs = bot_instance._pairs_cache[:100] if bot_instance._pairs_cache else []
        return jsonify({"pairs": pairs}), 200
    except Exception:
        return jsonify({"pairs": []}), 200


def run_trading_bot():
    global bot_instance

    async def start_bot():
        global bot_instance

        from src.zeus.core.bot import ZeusBot

        print("=" * 60)
        print("  ZEUS AUTONOMOUS TRADING BOT")
        print("  Raspberry Pi 5 Edition")
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
            bot_instance = ZeusBot(kraken_key=kraken_key,
                                   kraken_secret=kraken_secret,
                                   telegram_token=telegram_token,
                                   telegram_chat_id=telegram_chat_id,
                                   mode=mode)
            await bot_instance.run_forever()
        except Exception as e:
            with bot_lock:
                bot_status["error"] = str(e)
                bot_status["running"] = False
            print(f"Bot error: {e}")
            import traceback
            traceback.print_exc()

    asyncio.run(start_bot())


def run_server():
    port = int(os.environ.get("PORT", 5000))
    host = "0.0.0.0"

    print(f"[SERVER] Starting server on {host}:{port}")

    try:
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
            'bind': f'{host}:{port}',
            'workers': 1,
            'threads': 4,
            'timeout': 120,
            'accesslog': '-',
            'errorlog': '-',
            'loglevel': 'info'
        }

        print("[SERVER] Using Gunicorn production server")
        GunicornApp(app, options).run()
    except ImportError:
        print(
            "[SERVER] Gunicorn not available, using Flask development server")
        print(
            "[SERVER] Install gunicorn for production use: pip install gunicorn"
        )
        app.run(host=host, port=port, threaded=True, debug=False)


if __name__ == "__main__":
    # Create required directories
    Path("data").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    print("[ZEUS] Starting Zeus Trading Bot...")
    print(f"[ZEUS] Project root: {PROJECT_ROOT}")

    # Start trading bot in background thread
    bot_thread = threading.Thread(target=run_trading_bot, daemon=True)
    bot_thread.start()
    print("[BOT] Trading bot started in background thread")

    # Start web server
    run_server()
