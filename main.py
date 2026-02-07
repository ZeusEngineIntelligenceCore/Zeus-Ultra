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
from pathlib import Path
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
    return render_template("index.html",
                           user=current_user,
                           bot_status=bot_status,
                           current_time=format_la_time())


@app.route("/health")
def health():
    return {"status": "ok", "time": format_la_time()}, 200


@app.route("/api")
def api_root():
    return {"status": "ok", "version": "2.0", "time": format_la_time()}, 200


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
                at = state.get("active_trades", [])
                status_copy["active_trades"] = len(at) if isinstance(at, (list, dict)) else 0
                status_copy["candidates_count"] = len(
                    state.get("candidates", []))
                status_copy["total_profit"] = stats.get("total_pnl", 0)
                status_copy["cycle_count"] = state.get("cycle_count", 0)
                status_copy["wins"] = stats.get("wins", 0)
                status_copy["losses"] = stats.get("losses", 0)
                status_copy["mode"] = stats.get(
                    "mode", status_copy.get("mode", "UNKNOWN"))
                cfg = state.get("config", {})
                status_copy["bot_config"] = {
                    "strategy": cfg.get("strategy_mode", "day_trade"),
                    "min_confidence": cfg.get("min_confidence", 70),
                    "min_profit": cfg.get("min_profit_target", 3.0),
                    "max_positions": cfg.get("max_open_positions", 100),
                    "scan_interval": cfg.get("scan_interval", 300)
                }
                if hasattr(bot_instance, 'alt_data') and bot_instance.alt_data:
                    fg = bot_instance.alt_data.get_fear_greed()
                    if fg:
                        status_copy["fear_greed"] = {
                            "value": fg.value,
                            "classification": fg.classification
                        }
                state_loaded = True
            except Exception as e:
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
                    at = data.get("active_trades", {})
                    status_copy["active_trades"] = len(at) if isinstance(at, (list, dict)) else 0
                    status_copy["total_profit"] = round(
                        data.get("total_pnl", 0), 2)
                    status_copy["wins"] = data.get("wins", 0)
                    status_copy["losses"] = data.get("losses", 0)
                    config = data.get("config", {})
                    status_copy["mode"] = config.get(
                        "mode", status_copy.get("mode", "UNKNOWN"))
                    status_copy["bot_config"] = {
                        "strategy": config.get("strategy_mode", "day_trade"),
                        "min_confidence": config.get("min_confidence", 70),
                        "min_profit": config.get("min_profit_target", 3.0),
                        "max_positions": config.get("max_open_positions", 100),
                        "scan_interval": config.get("scan_interval", 300)
                    }
                    fg_data = data.get("fear_greed")
                    if fg_data:
                        status_copy["fear_greed"] = fg_data
            except Exception:
                pass

    wins = status_copy.get("wins", 0)
    losses = status_copy.get("losses", 0)
    if wins + losses > 0:
        status_copy["win_rate"] = round(wins / (wins + losses) * 100, 1)
    else:
        status_copy["win_rate"] = 0

    return jsonify(status_copy), 200


@app.route("/api/trades")
@require_login
def api_trades():
    trades = []
    def _parse_trade(trade, trade_id=None):
        return {
            "id": trade_id or trade.get("trade_id", trade.get("symbol", "unknown")),
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
        }
    try:
        if bot_instance:
            state = bot_instance.get_status()
            at = state.get("active_trades", [])
            if isinstance(at, list):
                for trade in at:
                    if isinstance(trade, dict):
                        trades.append(_parse_trade(trade))
            elif isinstance(at, dict):
                for trade_id, trade in at.items():
                    trades.append(_parse_trade(trade, trade_id))
        if not trades:
            state_file = Path("data/bot_state.json")
            if state_file.exists():
                with open(state_file, 'r') as f:
                    data = json.load(f)
                at = data.get("active_trades", {})
                if isinstance(at, list):
                    for trade in at:
                        if isinstance(trade, dict):
                            trades.append(_parse_trade(trade))
                elif isinstance(at, dict):
                    for trade_id, trade in at.items():
                        trades.append(_parse_trade(trade, trade_id))
        return jsonify({"trades": trades, "count": len(trades)}), 200
    except Exception as e:
        return jsonify({"trades": [], "error": str(e)}), 200


@app.route("/api/holdings")
@require_login
def api_holdings():
    holdings_data = {}
    active_trades = {}
    try:
        state_file = Path("data/bot_state.json")
        file_data = {}
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    file_data = json.load(f)
            except Exception:
                pass
        if bot_instance:
            state = bot_instance.get_status()
            holdings_data = state.get("holdings", {})
            at = state.get("active_trades", [])
            if isinstance(at, list):
                active_trades = {t.get("symbol", f"t{i}"): t for i, t in enumerate(at) if isinstance(t, dict)}
            elif isinstance(at, dict):
                active_trades = at
        if not holdings_data:
            holdings_data = file_data.get("holdings", {})
            if not active_trades:
                active_trades = file_data.get("active_trades", {})
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

            entry_price = 0.0
            if price == 0:
                for trade_id, trade in active_trades.items():
                    trade_symbol = trade.get("symbol", "")
                    if trade_symbol.startswith(
                            symbol) or trade_symbol.startswith(clean_symbol):
                        entry_price = trade.get("entry_price", 0)
                        price = entry_price
                        if price > 0:
                            break

            if price == 0 and file_data:
                saved_holdings_prices = file_data.get("holdings_prices", {})
                price = saved_holdings_prices.get(symbol, 0)

            value_usd = amount * price if price > 0 else 0
            pnl_pct = 0.0
            if entry_price > 0 and price > 0:
                pnl_pct = round((price - entry_price) / entry_price * 100, 2)

            holdings.append({
                "symbol": symbol,
                "amount": amount,
                "price": price,
                "value_usd": value_usd,
                "pnl_pct": pnl_pct
            })

        holdings.sort(key=lambda x: x.get("value_usd", 0), reverse=True)
        return jsonify({
            "holdings": holdings[:20],
            "count": len(holdings)
        }), 200
    except Exception as e:
        return jsonify({"holdings": [], "error": str(e)}), 200


recent_logs = []


@app.after_request
def add_cache_headers(response):
    if request.path.startswith('/api/') or request.path in ('/status', '/dashboard'):
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response


@app.route("/api/logs")
@require_login
def api_logs():
    global recent_logs
    log_paths = ["logs/zeus.log", "logs/trading.log"]
    for log_path in log_paths:
        try:
            log_file = Path(log_path)
            if log_file.exists() and log_file.stat().st_size > 0:
                import subprocess
                result = subprocess.run(["tail", "-50", log_path],
                                        capture_output=True,
                                        text=True,
                                        timeout=5)
                if result.stdout:
                    lines = result.stdout.strip().split('\n')
                    recent_logs = [l[:200] for l in lines[-30:]]
                    break
        except Exception:
            continue
    if not recent_logs:
        recent_logs = [f"[{format_la_time()}] Bot ready - waiting for manual start from dashboard"]
    return jsonify({"logs": recent_logs}), 200


@app.route("/api/candidates")
@require_login
def api_candidates():
    candidates = []
    try:
        if bot_instance and bot_instance.running:
            state = bot_instance.get_status()
            candidates = state.get("candidates", [])[:20]
        if not candidates:
            state_file = Path("data/bot_state.json")
            if state_file.exists():
                with open(state_file, 'r') as f:
                    data = json.load(f)
                candidates = data.get("candidates", [])[:20]
        return jsonify({
            "candidates": candidates,
            "count": len(candidates)
        }), 200
    except Exception as e:
        return jsonify({"candidates": [], "error": str(e)}), 200


@app.route("/api/ml/insights")
@require_login
def api_ml_insights():
    result = {"basic_ml": {}, "advanced_ml": {}}
    try:
        if bot_instance and bot_instance.running:
            result["basic_ml"] = bot_instance.ml_engine.get_global_insights()
            result["advanced_ml"] = bot_instance.advanced_ml.get_learning_summary()
        else:
            ml_file = Path("data/ml_learning.json")
            if ml_file.exists():
                with open(ml_file, 'r') as f:
                    result["basic_ml"] = json.load(f)
            adv_file = Path("data/advanced_ml.json")
            if adv_file.exists():
                with open(adv_file, 'r') as f:
                    result["advanced_ml"] = json.load(f)
    except Exception as e:
        result["error"] = str(e)
    return jsonify(result), 200


@app.route("/api/ml/telegram")
@require_login
def api_ml_telegram():
    try:
        tg_file = Path("data/telegram_learning.json")
        if tg_file.exists():
            with open(tg_file, 'r') as f:
                data = json.load(f)
            preferred = data.get("preferred_symbols", {})
            top_symbols = sorted(preferred.items(), key=lambda x: x[1], reverse=True)[:10]
            patterns = data.get("successful_trade_patterns", [])
            total_pnl = sum(p.get("pnl_pct", 0) for p in patterns)
            win_count = sum(1 for p in patterns if p.get("pnl_pct", 0) > 0)
            return jsonify({
                "status": "active",
                "top_symbols": dict(top_symbols),
                "avoided_symbols": data.get("avoided_symbols", []),
                "preferred_hours_utc": data.get("preferred_hours", []),
                "min_profit_threshold": data.get("min_profit_threshold", 3.0),
                "optimal_hold_mins": data.get("preferred_hold_duration_mins", 60),
                "confidence_threshold": data.get("learned_confidence_threshold", 70.0),
                "best_strategies": data.get("best_performing_strategies", []),
                "total_trades_learned": len(patterns),
                "total_pnl_learned": round(total_pnl, 2),
                "win_count": win_count,
                "loss_count": len(patterns) - win_count,
                "total_alerts_sent": data.get("total_alerts_sent", 0),
                "last_updated": data.get("last_updated", "Never"),
                "feedback": {
                    "positive": data.get("positive_feedback_count", 0),
                    "negative": data.get("negative_feedback_count", 0)
                }
            }), 200
        return jsonify({"status": "no_data", "message": "No ML learning data yet"}), 200
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 200


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
@require_login
def dashboard():
    return render_template("dashboard.html",
                           user=current_user,
                           bot_status=bot_status,
                           current_time=format_la_time())


@app.route("/api/analyze/<symbol>")
@require_login
def api_analyze_coin(symbol):
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    symbol = symbol.upper().strip()
    if not symbol.endswith("USD"):
        symbol = f"{symbol}USD"

    async def find_valid_pair(exchange, base_symbol):
        markets = await exchange.fetch_markets()
        if not markets:
            return base_symbol

        base = base_symbol.replace("USD", "")
        possible_names = [
            base_symbol,
            f"X{base}ZUSD",
            f"X{base}USD",
            f"{base}ZUSD",
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
        standalone_exchange = None
        try:
            if bot_instance:
                exchange = bot_instance.exchange
                prebreakout = bot_instance.prebreakout
            else:
                from src.zeus.exchanges.kraken import KrakenExchange
                from src.zeus.indicators.prebreakout_detector import PreBreakoutDetector
                kraken_key = os.environ.get("KRAKEN_API_KEY", "")
                kraken_secret = os.environ.get("KRAKEN_API_SECRET", "")
                standalone_exchange = KrakenExchange(api_key=kraken_key, api_secret=kraken_secret)
                await standalone_exchange.connect()
                exchange = standalone_exchange
                prebreakout = PreBreakoutDetector()
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
                return {
                    "error":
                    f"Could not fetch data for {symbol}. Make sure it's a valid Kraken trading pair."
                }

            valid_analyses = {
                k: v
                for k, v in all_analysis.items()
                if "error" not in v and v.get("current_price", 0) > 0
            }

            if not valid_analyses:
                error_msgs = [
                    f"{k}: {v.get('error', 'No data')}"
                    for k, v in all_analysis.items() if "error" in v
                ]
                return {
                    "error":
                    f"Failed to fetch data for {symbol}. Kraken API may be temporarily unavailable. "
                    + (error_msgs[0] if error_msgs else "")
                }

            primary_tf = valid_analyses.get("15m") or valid_analyses.get(
                "1h") or list(valid_analyses.values())[0]

            return {
                "symbol": symbol,
                "timestamp": format_la_time(),
                "primary_analysis": primary_tf,
                "timeframes": valid_analyses,
                "kpi_count": primary_tf.get("kpi_count", 30),
                "success": True
            }
        except Exception as e:
            return {"error": str(e)}
        finally:
            if standalone_exchange:
                try:
                    await standalone_exchange.disconnect()
                except Exception:
                    pass

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(run_analysis())
        loop.close()
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/analyze/<symbol>/telegram", methods=["POST"])
@require_login
def api_analyze_to_telegram(symbol):
    if not bot_instance:
        return jsonify({"error": "Bot not running"}), 503

    import asyncio

    symbol = symbol.upper().strip()
    if not symbol.endswith("USD"):
        symbol = f"{symbol}USD"

    async def send_analysis():
        try:
            telegram = bot_instance.telegram
            exchange = bot_instance.exchange
            prebreakout = bot_instance.prebreakout

            ohlcv = await exchange.fetch_ohlcv(symbol, "15m", limit=500)
            if not ohlcv or len(ohlcv) < 50:
                return {"error": f"Insufficient data for {symbol}"}

            high = [c.high for c in ohlcv]
            low = [c.low for c in ohlcv]
            close = [c.close for c in ohlcv]
            volume = [c.volume for c in ohlcv]

            analysis = await prebreakout.analyze(symbol, high, low, close,
                                                 volume)

            score = analysis.get("prebreakout_score", 0)
            stage = analysis.get("stage", "UNKNOWN")
            confidence = analysis.get("confidence", 0)
            price = analysis.get("current_price", 0)
            features = analysis.get("features", {})
            reasons = analysis.get("reasons", [])

            stage_emoji = {
                "BREAKOUT": "🚀",
                "LATE_PRE-BREAKOUT": "🔥",
                "PRE-BREAKOUT": "⚡",
                "EARLY_SETUP": "📊",
                "ACCUMULATION": "🔄",
                "DORMANT": "💤"
            }.get(stage, "📈")

            msg_lines = [
                f"🔬 <b>ZEUS COIN ANALYSIS</b>", f"━━━━━━━━━━━━━━━━━━━━", f"",
                f"<b>Symbol:</b> {symbol}", f"<b>Price:</b> ${price:.8f}"
                if price < 1 else f"<b>Price:</b> ${price:.4f}",
                f"<b>Stage:</b> {stage_emoji} {stage}",
                f"<b>Score:</b> {score:.1f}/100",
                f"<b>Confidence:</b> {confidence:.1f}%", f"",
                f"━━━ <b>30 KPI ANALYSIS</b> ━━━"
            ]

            kpi_items = [("RSI", features.get("rsi", 0)),
                         ("Momentum", features.get("momentum_cf", 0)),
                         ("Volume Spike", features.get("vol_spike", 0)),
                         ("Pressure", features.get("pressure", 0)),
                         ("Microtrend", features.get("microtrend", 0)),
                         ("Impulse", features.get("impulse", 0)),
                         ("Squeeze", features.get("squeeze", 0)),
                         ("ADX Strength", features.get("adx_strength", 0)),
                         ("Supertrend", features.get("supertrend_conf", 0)),
                         ("Aroon", features.get("aroon_signal", 0)),
                         ("Vortex", features.get("vortex_signal", 0)),
                         ("Williams %R", features.get("williams_r", 0)),
                         ("Stoch RSI", features.get("stoch_rsi", 0)),
                         ("MFI", features.get("mfi_signal", 0)),
                         ("OBV Trend", features.get("obv_trend", 0)),
                         ("CCI", features.get("cci_signal", 0)),
                         ("Pivot Dist", features.get("pivot_distance", 0)),
                         ("Fib Level", features.get("fibonacci_level", 0)),
                         ("Parabolic SAR", features.get("parabolic_sar", 0)),
                         ("Elder Power", features.get("elder_power", 0)),
                         ("Ultimate Osc", features.get("ultimate_osc", 0)),
                         ("Choppiness", features.get("choppiness", 0)),
                         ("Klinger", features.get("klinger_signal", 0)),
                         ("Donchian", features.get("donchian_position", 0)),
                         ("LinReg", features.get("linreg_trend", 0)),
                         ("Acceleration", features.get("accel", 0)),
                         ("Consistency", features.get("consistency", 0)),
                         ("Liquidity", features.get("liquidity", 0)),
                         ("Vol Anomaly", features.get("anomaly_vol", 0)),
                         ("Candle Proj", features.get("candle_proj", 0))]

            for name, val in kpi_items:
                bar = "█" * int(val * 10) + "░" * (10 - int(val * 10))
                pct = val * 100
                emoji = "🟢" if pct >= 60 else "🟡" if pct >= 40 else "🔴"
                msg_lines.append(f"{emoji} {name}: {bar} {pct:.0f}%")

            if reasons:
                msg_lines.append("")
                msg_lines.append("━━━ <b>SIGNALS</b> ━━━")
                for reason in reasons[:5]:
                    msg_lines.append(f"✅ {reason}")

            msg_lines.extend([
                "", f"━━━ <b>TRADE LEVELS</b> ━━━",
                f"🎯 Entry: ${analysis.get('buy_anchor', 0):.8f}" if price < 1
                else f"🎯 Entry: ${analysis.get('buy_anchor', 0):.4f}",
                f"🛑 Stop Loss: ${analysis.get('stop_loss', 0):.8f}" if price
                < 1 else f"🛑 Stop Loss: ${analysis.get('stop_loss', 0):.4f}",
                f"💰 Take Profit: ${analysis.get('take_profit', 0):.8f}"
                if price < 1 else
                f"💰 Take Profit: ${analysis.get('take_profit', 0):.4f}", "",
                f"⏰ {format_la_time()}"
            ])

            message = "\n".join(msg_lines)
            await telegram.send_message(message,
                                        alert_type="analysis",
                                        urgent=True)

            return {"success": True, "message": "Analysis sent to Telegram"}
        except Exception as e:
            return {"error": str(e)}

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(send_analysis())
        loop.close()
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/miniapp")
def miniapp():
    return render_template("miniapp.html")


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
        'workers': 1,
        'threads': 4,
        'timeout': 120,
        'accesslog': '-',
        'errorlog': '-',
        'loglevel': 'warning'
    }

    GunicornApp(app, options).run()


if __name__ == "__main__":
    print("[BOT] Bot ready - toggle ON from dashboard to start trading")
    bot_status["running"] = False
    
    run_server()
