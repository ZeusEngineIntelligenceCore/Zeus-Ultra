#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZEUS ULTRA • CAT-9 HYDRA-D CORE (2025)
FULL UNIFIED MEGA VERSION — FULLY PATCHED
=========================================
Tactical engine for pre-breakout scoring and laddered target generation.

NOTES (REPAIRED):
- Fixed main() logic (scan_all vs scan_specific) + removed undefined pairs bug
- Removed duplicate httpx import blocks and consolidated fallback
- Hydra fuse now uses deterministic timeframe ordering (VALID_TFS)
- Added guardrails for missing/empty OHLC data
- Preserved your CAT9Ultra engine logic (minimal invasive repairs)
"""

from __future__ import annotations

import asyncio
import json
import math
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# ============================================================
# PATHS + SECRETS (optional)
# ============================================================

CONFIG_DIR = Path.cwd().parent / "config" if "zeus" in str(Path.cwd()) else Path.cwd() / "config"
SECRETS_PATH = CONFIG_DIR / "secrets.json"


def load_secrets() -> Dict[str, Any]:
    """Loads secrets.json securely (optional for public Kraken endpoints)."""
    if not SECRETS_PATH.exists():
        raise FileNotFoundError(f"Missing secrets file at: {SECRETS_PATH}")
    return json.loads(SECRETS_PATH.read_text())


# Optional secrets (public endpoints do not require them)
KRAKEN_KEY = ""
KRAKEN_SECRET = ""
TELEGRAM_TOKEN = ""

try:
    s = load_secrets()
    KRAKEN_KEY = s.get("kraken", {}).get("api_key", "") or ""
    KRAKEN_SECRET = s.get("kraken", {}).get("api_secret", "") or ""
    TELEGRAM_TOKEN = s.get("telegram", {}).get("bot_token", "") or ""
    print("[CAT-9] Secrets loaded successfully.")
except Exception as e:
    print(f"[CAT-9] Secrets not loaded (ok for public scan): {e}")


# ============================================================
# GLOBAL CONFIG
# ============================================================

API_BASE = "https://api.kraken.com/0/public"
VALID_TFS = [5, 15, 60]

MIN_BARS_REQUIRED = 30

DEFAULT_STABLE_BASES = {
    "USDT", "USDC", "DAI", "TUSD", "USDS", "BUSD", "PYUSD",
    "EURT", "GBPQ", "UST", "USDQ"
}

DISCOVERY_LOG = Path("ultra/cat9_outputs/discovery_log.txt")
ALERT_LOG = Path("ultra/cat9_outputs/scan_log.txt")


# ============================================================
# PYTHONISTA MODE OVERRIDE (kept)
# ============================================================

class Args:
    scan_all_usd = True
    pairs = ""
    bars = 200
    top = 15
    exclude_stables = True
    outdir = "ultra/cat9_outputs"


# ============================================================
# HTTP CLIENT (httpx if available, fallback if not)
# ============================================================

try:
    import httpx  # type: ignore
    HAVE_HTTPX = True
except Exception:
    HAVE_HTTPX = False

    class httpx:  # type: ignore
        """Minimal Pythonista fallback for httpx.AsyncClient"""

        class Response:
            def __init__(self, raw: str):
                self._raw = raw

            def json(self):
                try:
                    return json.loads(self._raw)
                except Exception:
                    return {}

        class AsyncClient:
            def __init__(self, base_url: Optional[str] = None, **kwargs):
                self.base_url = base_url or ""

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, url: str, params: Optional[dict] = None):
                if params:
                    full = f"{self.base_url}{url}?{urllib.parse.urlencode(params)}"
                else:
                    full = f"{self.base_url}{url}"
                with urllib.request.urlopen(full) as r:
                    return httpx.Response(r.read().decode("utf-8"))

        class Timeout:
            def __init__(self, *a, **k): ...

        class Limits:
            def __init__(self, *a, **k): ...

        class AsyncHTTPTransport:
            def __init__(self, retries: int = 0):
                self.retries = retries


def pythonista_httpx_client() -> "httpx.AsyncClient":
    """Safe client usable in Pythonista iOS."""
    if HAVE_HTTPX:
        limits = httpx.Limits(max_connections=40, max_keepalive_connections=20)
        timeout = httpx.Timeout(20, connect=20, read=20)
        transport = httpx.AsyncHTTPTransport(retries=2)
        headers = {
            "User-Agent": "Zeus-CAT9/2025",
            "Accept": "application/json",
        }
        return httpx.AsyncClient(
            base_url=API_BASE,
            headers=headers,
            limits=limits,
            timeout=timeout,
            transport=transport,
        )
    return httpx.AsyncClient(base_url=API_BASE)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def as_list(x: Any) -> List[float]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    return [float(x)]


def pct_change(seq: Sequence[float]) -> List[float]:
    seq = as_list(seq)
    out = []
    for i in range(1, len(seq)):
        prev = seq[i - 1] if seq[i - 1] else 1.0
        out.append((seq[i] - seq[i - 1]) / abs(prev))
    return out


def safe_mean(vals, default=0.0):
    vals = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
    return sum(vals) / len(vals) if vals else default


def safe_std(vals, default=0.0):
    vals = as_list(vals)
    if len(vals) < 2:
        return default
    m = safe_mean(vals)
    v = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return math.sqrt(v)


def clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def linear_slope(seq: Sequence[float]) -> float:
    seq = as_list(seq)
    n = len(seq)
    if n < 2:
        return 0.0
    x = list(range(n))
    mx = (n - 1) / 2
    my = safe_mean(seq)
    num = sum((x[i] - mx) * (seq[i] - my) for i in range(n))
    den = sum((xi - mx) ** 2 for xi in x) or 1.0
    return num / den


def tanh01(x: float) -> float:
    return (math.tanh(x) + 1) / 2


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x)) if x < 60 else 1.0


# ============================================================
# FETCH + DISCOVERY
# ============================================================

async def fetch_json(client, endpoint: str, params: Optional[dict] = None) -> Dict[str, Any]:
    r = await client.get(endpoint, params=params)
    try:
        j = r.json()
    except Exception:
        return {}
    if isinstance(j, dict) and j.get("error"):
        return {}
    return j.get("result", {}) or {}


async def discover_usd_pairs(client, exclude_stables: bool = True) -> List[str]:
    """Bulletproof Kraken USD pair discovery."""
    raw = await fetch_json(client, "/AssetPairs")
    out: List[str] = []

    for name, info in raw.items():
        ws = str(info.get("wsname", "")).upper()
        quote = str(info.get("quote", "")).upper()
        base = (
            str(info.get("base", ""))
            .replace("X", "")
            .replace("Z", "")
            .upper()
        )

        is_usd = (
            quote.endswith("USD")
            or ws.endswith("/USD")
            or name.upper().endswith("USD")
        )
        if not is_usd:
            continue

        if exclude_stables and base in DEFAULT_STABLE_BASES:
            continue

        out.append(name)

    out = sorted(set(out))

    DISCOVERY_LOG.parent.mkdir(parents=True, exist_ok=True)
    with DISCOVERY_LOG.open("w") as f:
        for p in out:
            f.write(p + "\n")

    print(f"[DISCOVERY] Found {len(out)} USD pairs")
    print(f"[LOG] → {DISCOVERY_LOG}")

    return out


async def fetch_ohlc(client, pair: str, interval: int, bars: int):
    raw = await fetch_json(client, "/OHLC", {"pair": pair, "interval": interval})
    if not raw:
        return [], [], []

    key = next((k for k in raw.keys() if k != "last"), None)
    if not key:
        return [], [], []

    rows = raw.get(key, []) or []
    rows = rows[-bars:]

    try:
        closes = [float(r[4]) for r in rows]
        vols = [float(r[6]) for r in rows]
    except Exception:
        return [], [], []

    liq = [c * v for c, v in zip(closes, vols)]
    return closes, vols, liq


# ============================================================
# ENGINE CONFIG
# ============================================================

@dataclass
class EngineConfig:
    rsi_period: int = 14
    poly_degree: int = 2
    spike_cap: float = 0.25
    pressure_cap: float = 1.8
    impulse_cap: float = 6.0
    liquidity_scale: float = 0.6

    break_pre: float = 70.0
    break_out: float = 85.0

    ladder_tiers: int = 3
    ladder_step_atr: float = 0.6
    ladder_sell_step: float = 0.8


# ============================================================
# CAT-9 ULTRA ENGINE
# ============================================================

class CAT9Ultra:
    def __init__(self, cfg: Optional[EngineConfig] = None):
        self.cfg = cfg or EngineConfig()

    async def predictive_rsi(self, prices):
        prices = as_list(prices)
        if len(prices) < 4:
            return 50.0
        gains = []
        losses = []
        for i in range(1, len(prices)):
            d = prices[i] - prices[i - 1]
            gains.append(max(0.0, d))
            losses.append(max(0.0, -d))
        p = self.cfg.rsi_period
        ag = safe_mean(gains[-p:])
        al = safe_mean(losses[-p:]) or 1e-9
        rsi = 100.0 - 100.0 / (1.0 + ag / al)
        return clamp(rsi, 0.0, 100.0)

    async def momentum_cf(self, prices):
        prices = as_list(prices)
        if len(prices) < 4:
            return 0.5
        slope = linear_slope(prices[-12:])
        return clip01(tanh01(slope * 6))

    async def vol_spike(self, prices):
        prices = as_list(prices)
        if len(prices) < 4:
            return 0.0
        lo, hi = min(prices), max(prices)
        med = sorted(prices)[len(prices) // 2]
        raw = (hi - lo) / abs(med or 1e-9)
        return clip01(raw / self.cfg.spike_cap)

    async def pressure(self, vols, prices):
        vols = as_list(vols)
        prices = as_list(prices)
        if len(prices) < 4:
            return 0.0
        rets = pct_change(prices)
        last = rets[-1] if rets else 0.0
        med = sorted(vols)[len(vols) // 2] if vols else 1e-9
        vol_ratio = safe_mean(vols[-3:]) / (med or 1e-9)
        raw = abs(last) * vol_ratio
        return clip01(raw / self.cfg.pressure_cap)

    async def microtrend(self, prices):
        prices = as_list(prices)
        return clip01(tanh01(linear_slope(prices[-15:]) * 8))

    async def accel(self, prices):
        prices = as_list(prices)
        rets = pct_change(prices)
        if len(rets) < 4:
            return 0.5
        dif = [rets[i] - rets[i - 1] for i in range(1, len(rets))]
        jerk = [dif[i] - dif[i - 1] for i in range(1, len(dif))]
        return clip01(tanh01(safe_mean(jerk) * 20))

    async def anomaly_vol(self, vols):
        vols = as_list(vols)
        if len(vols) < 4:
            return 0.0
        med = sorted(vols)[len(vols) // 2]
        sd = safe_std(vols) or 1e-9
        z = (vols[-1] - med) / sd
        return clip01(sigmoid(z / 2))

    async def candle_proj(self, prices):
        prices = as_list(prices)
        if len(prices) < 4:
            return 0.5
        bodies = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
        slope = linear_slope(bodies[-8:])
        wick = (max(prices) - min(prices)) - safe_mean(bodies)
        denom = safe_mean(bodies or [1e-9]) + 1.0
        return clip01(tanh01(slope * 5) * tanh01(wick / denom))

    async def consistency(self, prices):
        prices = as_list(prices)
        if len(prices) < 8:
            return 0.5
        seg = max(4, len(prices) // 4)
        slopes = []
        for i in range(seg, len(prices) + 1, seg):
            slopes.append(linear_slope(prices[max(0, i - seg):i]))
        m = safe_mean(slopes)
        sd = safe_std(slopes)
        return clip01(1.0 - (sd / (abs(m) + 1e-9)))

    async def impulse(self, prices):
        prices = as_list(prices)
        rets = pct_change(prices)
        if len(rets) < 4:
            return 0.0
        m = safe_mean(rets)
        sd = safe_std(rets) or 1e-9
        z = abs((rets[-1] - m) / sd)
        return clip01(sigmoid(clamp(z, 0.0, self.cfg.impulse_cap) - 1.0))

    async def liquidity_shift(self, liq):
        liq = as_list(liq)
        if len(liq) < 4:
            return 0.5
        med = sorted(liq)[len(liq) // 2]
        ratio = (liq[-1] / (med or 1e-9)) - 1.0
        return clip01(tanh01(ratio / self.cfg.liquidity_scale))

    async def liftoff(self, list_of_tf_pre):
        vals = [v for v in as_list(list_of_tf_pre) if 0.0 <= v <= 1.0]
        return clip01(safe_mean(vals, 0.0))

    def atr(self, prices):
        prices = as_list(prices)
        if len(prices) < 3:
            return 0.0
        rets = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
        return safe_mean(rets[-14:], safe_mean(rets))

    def stage_label(self, preb, feats):
        if preb >= self.cfg.break_out:
            if feats.get("impulse", 0.0) >= 0.8 or feats.get("pressure", 0.0) >= 0.7:
                return "post-breakout"
        if preb >= self.cfg.break_pre:
            return "pre-breakout"
        return "neutral"

    def build_ladders(self, prices, feats):
        prices = as_list(prices)
        last = prices[-1]
        win = max(10, len(prices) // 3)
        rec_lo = min(prices[-win:])
        rec_hi = max(prices[-win:])
        atr_val = self.atr(prices) or (last * 0.002)

        buy_anchor = rec_lo + (0.3 * feats["microtrend"] + 0.2 * feats["momentum_cf"]) * atr_val
        sell_anchor = rec_hi + (0.5 * feats["impulse"] + 0.3 * feats["candle_proj"]) * atr_val

        buys = {
            f"tier{t}": round(buy_anchor - (t - 1) * self.cfg.ladder_step_atr * atr_val, 8)
            for t in range(1, self.cfg.ladder_tiers + 1)
        }
        sells = {
            f"tier{t}": round(sell_anchor + (t - 1) * self.cfg.ladder_sell_step * atr_val, 8)
            for t in range(1, self.cfg.ladder_tiers + 1)
        }

        return buys, sells, round(buy_anchor, 8), round(sell_anchor, 8)

    async def run(self, symbol, prices, vols, liq, tf_scores):
        prices = as_list(prices)
        vols = as_list(vols)
        liq = as_list(liq)

        tasks = await asyncio.gather(
            self.predictive_rsi(prices),
            self.momentum_cf(prices),
            self.vol_spike(prices),
            self.pressure(vols, prices),
            self.microtrend(prices),
            self.accel(prices),
            self.anomaly_vol(vols),
            self.candle_proj(prices),
            self.consistency(prices),
            self.impulse(prices),
            self.liquidity_shift(liq),
            self.liftoff(tf_scores),
        )

        names = [
            "rsi", "momentum_cf", "vol_spike", "pressure", "microtrend",
            "accel", "anomaly_vol", "candle_proj", "consistency",
            "impulse", "liquidity", "liftoff"
        ]
        feats = {k: round(v, 4) for k, v in zip(names, tasks)}

        raw = sum([
            feats["rsi"] * 0.14,
            feats["momentum_cf"] * 0.10,
            feats["vol_spike"] * 0.10,
            feats["pressure"] * 0.12,
            feats["microtrend"] * 0.08,
            feats["accel"] * 0.06,
            feats["anomaly_vol"] * 0.05,
            feats["candle_proj"] * 0.07,
            feats["consistency"] * 0.08,
            feats["impulse"] * 0.08,
            feats["liquidity"] * 0.06,
            feats["liftoff"] * 0.06,
        ])

        preb = round(clip01(raw) * 100.0, 4)
        prob = round(math.tanh(preb / 85.0), 4)
        enh = round(1.0 - math.exp(-(preb) / 85.0), 4)

        stage = self.stage_label(preb, feats)
        buy_lad, sell_lad, b_anchor, s_anchor = self.build_ladders(prices, feats)

        return {
            "symbol": symbol,
            "stage": stage,
            "prebreakout_score": preb,
            "breakout_prob_pre": prob,
            "breakout_prob_enhanced": enh,
            "buy_anchor": b_anchor,
            "sell_anchor": s_anchor,
            "buy_ladder": buy_lad,
            "sell_ladder": sell_lad,
            **feats,
        }


# ============================================================
# HYDRA MULTI-TF FUSION (REPAIRED ORDERING)
# ============================================================

def hydra_fuse(tf_data: Dict[int, Dict[str, Any]]):
    # Deterministic order: use VALID_TFS and only include those present
    tfs = [t for t in VALID_TFS if t in tf_data]
    if not tfs:
        return {}

    scores = [tf_data[t]["prebreakout_score"] / 100.0 for t in tfs]
    align = safe_mean([tf_data[t]["microtrend"] for t in tfs])
    mom = safe_mean([tf_data[t]["momentum_cf"] for t in tfs])

    fusion = clip01(math.tanh(0.4 * safe_mean(scores)))
    final = clip01(fusion * (0.7 + 0.3 * ((align + mom) / 2.0)))

    core = tf_data.get(15) or tf_data[tfs[len(tfs) // 2]]

    return {
        "prebreakout_score": round(final * 100.0, 4),
        "breakout_prob_pre": round(safe_mean([tf_data[t]["breakout_prob_pre"] for t in tfs]), 4),
        "breakout_prob_enhanced": round(safe_mean([tf_data[t]["breakout_prob_enhanced"] for t in tfs]), 4),
        "buy_anchor": core["buy_anchor"],
        "sell_anchor": core["sell_anchor"],
        "buy_ladder": core["buy_ladder"],
        "sell_ladder": core["sell_ladder"],
        **{k: core[k] for k in [
            "rsi", "momentum_cf", "vol_spike", "pressure", "microtrend", "accel",
            "anomaly_vol", "candle_proj", "consistency", "impulse", "liquidity", "liftoff"
        ]},
    }


# ============================================================
# AUTO-ALERT SYSTEM
# ============================================================

def auto_alert(result: Dict[str, Any]):
    if result.get("stage") != "pre-breakout":
        return

    msg = (
        f"\n🔥🔥🔥 PRE-BREAKOUT DETECTED! 🔥🔥🔥\n"
        f"PAIR: {result.get('symbol')}\n"
        f"SCORE: {result.get('prebreakout_score')}%\n"
        f"PROB PRE: {result.get('breakout_prob_pre')}\n"
        f"BUY: {result.get('buy_anchor')}\n"
        f"SELL: {result.get('sell_anchor')}\n"
        + "-" * 40 + "\n"
    )
    print(msg)

    ALERT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with ALERT_LOG.open("a") as f:
        f.write(msg)


# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

async def analyze_pair(client, engine: CAT9Ultra, pair: str, bars: int):
    tf_results: Dict[int, Dict[str, Any]] = {}

    for tf in VALID_TFS:
        c, v, l = await fetch_ohlc(client, pair, tf, bars)

        if len(c) < MIN_BARS_REQUIRED:
            print(f"[WARN] Insufficient data for {pair}@{tf}m. Got {len(c)}, need {MIN_BARS_REQUIRED}")
            return None

        tf_results[tf] = await engine.run(pair, c, v, l, [])

    fused = hydra_fuse(tf_results)
    if not fused:
        return None

    fused["symbol"] = pair
    fused["stage"] = engine.stage_label(fused["prebreakout_score"], fused)
    fused["timeframes"] = VALID_TFS
    return fused


async def scan_all(engine: CAT9Ultra, bars: int, top_n: int, exclude_stables: bool):
    bars = max(bars, MIN_BARS_REQUIRED)

    async with pythonista_httpx_client() as client:
        pairs = await discover_usd_pairs(client, exclude_stables)

        results: List[Dict[str, Any]] = []
        sem = asyncio.Semaphore(8)

        async def worker(p: str):
            async with sem:
                try:
                    r = await analyze_pair(client, engine, p, bars)
                    if r:
                        results.append(r)
                except Exception as e:
                    print(f"[ERROR] Failed to analyze {p}: {e}")

        await asyncio.gather(*(worker(p) for p in pairs))

    results.sort(key=lambda r: r["prebreakout_score"], reverse=True)
    return results[:top_n] if top_n else results


async def scan_specific(engine: CAT9Ultra, pairs: List[str], bars: int):
    bars = max(bars, MIN_BARS_REQUIRED)

    async with pythonista_httpx_client() as client:
        out: List[Dict[str, Any]] = []
        for p in pairs:
            try:
                r = await analyze_pair(client, engine, p, bars)
                if r:
                    out.append(r)
            except Exception as e:
                print(f"[ERROR] Failed to analyze {p}: {e}")

    out.sort(key=lambda r: r["prebreakout_score"], reverse=True)
    return out


# ============================================================
# OUTPUT + PRINT
# ============================================================

def print_board(results: List[Dict[str, Any]], top_n: int):
    if not results:
        print("No results.")
        return

    best = results[0]
    print("\n" + "=" * 72)
    print(f"🔥 BEST PRE-BREAKOUT → {best['symbol']} | {best['prebreakout_score']}%")
    print(f"BUY: {best['buy_anchor']} | SELL: {best['sell_anchor']}")
    print("=" * 72)

    show = results[:top_n] if top_n else results
    print("\n🔥 BREAKOUT LEADERBOARD")
    print("-" * 72)
    for i, r in enumerate(show, 1):
        auto_alert(r)
        print(
            f"{i:>2}. {r['symbol']:<10} {r['stage']:<12} "
            f"{r['prebreakout_score']:>6.2f}%  "
            f"BuyT1 {r['buy_ladder']['tier1']}  "
            f"SellT1 {r['sell_ladder']['tier1']}"
        )
    print("-" * 72)


def save_json(results: List[Dict[str, Any]], outdir: str):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "cat9_ranked.json"
    with path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {path}")
    return path


# ============================================================
# CLI
# ============================================================

def parse_args():
    # Pythonista forced auto mode (kept)
    a = Args()
    a.scan_all_usd = True
    a.pairs = ""
    return a


async def main():
    args = parse_args()
    eng = CAT9Ultra()

    # FIXED: proper branching (no undefined pairs, no overwriting results)
    if args.scan_all_usd:
        results = await scan_all(
            eng,
            args.bars,
            args.top,
            args.exclude_stables,
        )
    else:
        pairs = [p.strip() for p in str(args.pairs).split(",") if p.strip()]
        if not pairs:
            pairs = ["XBTUSD", "ETHUSD"]
        results = await scan_specific(eng, pairs, args.bars)

    print_board(results, args.top)
    save_json(results, args.outdir)


if __name__ == "__main__":
    asyncio.run(main())


# ============================================================
# MODULE-LEVEL SYNC RUNNER (compat)
# ============================================================

def run(*, bus=None, state=None):
    """Module-level runner for compatibility."""
    try:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                print("[SYNC RUNNER] WARNING: Async loop is running. Cannot start new main().")
                return {"error": "Async loop already running"}
        except RuntimeError:
            # No loop exists yet
            pass

        asyncio.run(main())
        return {"status": "success", "message": "CAT-9 analysis complete"}
    except Exception as e:
        return {"error": f"CAT-9 Sync Runner Failure: {e}"}
