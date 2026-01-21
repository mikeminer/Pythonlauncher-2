import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import asyncio
import json
import time
import threading
from collections import deque

import requests
import websockets

# Windows UI
import tkinter as tk
from tkinter import ttk
import winsound


# =========================
# CONFIG
# =========================
SYMBOL = "ethusdt"
SYMBOL_UP = SYMBOL.upper()

STREAM = (
    "wss://fstream.binance.com/stream?streams="
    f"{SYMBOL}@aggTrade/"
    f"{SYMBOL}@depth@100ms/"
    f"{SYMBOL}@forceOrder"
)

REST_BASE = "https://fapi.binance.com"
OI_ENDPOINT = f"{REST_BASE}/fapi/v1/openInterest"
PREMIUM_ENDPOINT = f"{REST_BASE}/fapi/v1/premiumIndex"

DEPTH_LEVELS = 20
REST_POLL_SEC = 10
PRINT_EVERY_SEC = 1.0

# Timeframes we always compute
TF_OPTIONS = {
    "5m": 5 * 60,
    "15m": 15 * 60,
    "30m": 30 * 60,
    "1h": 60 * 60,
    "2h": 2 * 60 * 60,
    "4h": 4 * 60 * 60,
}

# =========================
# ROBUST REST SESSION
# =========================
REST_TIMEOUT = 12
REST_MAX_RETRIES = 3

retry_cfg = Retry(
    total=REST_MAX_RETRIES,
    connect=REST_MAX_RETRIES,
    read=REST_MAX_RETRIES,
    backoff_factor=0.7,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    raise_on_status=False,
)

REST_SESSION = requests.Session()
REST_SESSION.mount("https://", HTTPAdapter(max_retries=retry_cfg))

# Dynamic thresholds per TF (RED/CLEAR)
TF_THRESHOLDS = {
    "5m":  {"RED": 8.5, "CLEAR": 5.5},
    "15m": {"RED": 7.5, "CLEAR": 4.8},
    "30m": {"RED": 7.0, "CLEAR": 4.5},
    "1h":  {"RED": 6.5, "CLEAR": 4.2},
    "2h":  {"RED": 6.0, "CLEAR": 4.0},
    "4h":  {"RED": 5.5, "CLEAR": 3.8},
}

# TF hierarchy rank (higher = more systemic)
TF_RANK = {
    "5m": 1,
    "15m": 2,
    "30m": 3,
    "1h": 4,
    "2h": 5,
    "4h": 6,
}

# Profiles (one click)
PROFILES = {
    "scalp": {
        "popup_cooldown": 60,
        "auto_clear": True,
        "force_popup_on_global_red": True,
    },
    "swing": {
        "popup_cooldown": 30,
        "auto_clear": True,
        "force_popup_on_global_red": True,
    },
    "night": {
        "popup_cooldown": 15,
        "auto_clear": False,   # sirena NON si spegne da sola
        "force_popup_on_global_red": True,
    },
}


# =========================
# USER CHOICES
# =========================
def choose_display_timeframe():
    print("\nSeleziona timeframe DISPLAY (quello che stampo a schermo):")
    keys = list(TF_OPTIONS.keys())
    for i, k in enumerate(keys, 1):
        print(f"  {i}) {k}")
    sel = input("Scelta (1-6) oppure scrivi es. 15m: ").strip().lower()

    if sel in TF_OPTIONS:
        return sel
    try:
        idx = int(sel)
        if 1 <= idx <= len(keys):
            return keys[idx - 1]
    except:
        pass

    print("Input non valido -> default 15m")
    return "15m"


def choose_profile():
    print("\nSeleziona profilo operativo:")
    print("  1) Scalp")
    print("  2) Swing")
    print("  3) Night shift")
    sel = input("Scelta (1-3): ").strip()

    if sel == "1":
        return "scalp"
    if sel == "3":
        return "night"
    return "swing"


# =========================
# STATE
# =========================
state_lock = threading.Lock()

# tick stores (pruned by max window = 4h)
MAX_WINDOW_SEC = TF_OPTIONS["4h"]

trades = deque()      # (ts_ms, qty, is_buyer_maker, price)
liq_events = deque()  # (ts_s, side, qty, price) side: "LONG"/"SHORT"

bids_cache = []
asks_cache = []
last_mid = None

oi_history = deque(maxlen=5000)       # (ts_s, oi)
funding_history = deque(maxlen=5000)  # (ts_s, funding, mark, index)

# alert / UI
alarm_active = False
popup_active = False
snooze_until = 0
last_popup_ts = 0
last_alert_mode = None

# Auto-switch: global highest TF currently RED
GLOBAL_RED_TF_RANK = 0
GLOBAL_RED_TF_LABEL = None


# =========================
# REST HELPERS
# =========================
def get_open_interest():
    r = REST_SESSION.get(
        OI_ENDPOINT,
        params={"symbol": SYMBOL_UP},
        timeout=REST_TIMEOUT,
    )
    r.raise_for_status()
    return float(r.json()["openInterest"])



def get_funding_mark_index():
    r = REST_SESSION.get(
        PREMIUM_ENDPOINT,
        params={"symbol": SYMBOL_UP},
        timeout=REST_TIMEOUT,
    )
    r.raise_for_status()
    j = r.json()
    return float(j["lastFundingRate"]), float(j["markPrice"]), float(j["indexPrice"])

    return funding, mark, index


# =========================
# METRICS
# =========================
def compute_mid(bids, asks):
    if not bids or not asks:
        return None
    try:
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        return (best_bid + best_ask) / 2.0
    except Exception:
        return None


def depth_imbalance(bids, asks, levels=DEPTH_LEVELS):
    try:
        b = sum(float(q) for _, q in bids[:levels])
        a = sum(float(q) for _, q in asks[:levels])
        if a + b == 0:
            return 0.0
        return (b - a) / (a + b)
    except Exception:
        return 0.0


def prune_ticks(now_s):
    """Keep only last MAX_WINDOW_SEC of data (trades + liquidations)."""
    trade_cutoff_ms = int((now_s - MAX_WINDOW_SEC) * 1000)
    liq_cutoff_s = now_s - MAX_WINDOW_SEC

    with state_lock:
        while trades and trades[0][0] < trade_cutoff_ms:
            trades.popleft()
        while liq_events and liq_events[0][0] < liq_cutoff_s:
            liq_events.popleft()


def cvd_window(now_ms, window_sec):
    cutoff = now_ms - window_sec * 1000
    buy = 0.0
    sell = 0.0

    with state_lock:
        snap = list(trades)

    # Iterate only relevant trades
    for ts, qty, is_buyer_maker, price in snap:
        if ts < cutoff:
            continue
        # m=True => taker sell
        if is_buyer_maker:
            sell += qty
        else:
            buy += qty
    return (buy - sell), buy, sell


def liq_stats(now_s, window_sec):
    cutoff = now_s - window_sec
    long_qty = 0.0
    short_qty = 0.0
    cnt = 0

    with state_lock:
        snap = list(liq_events)

    for ts_s, side, qty, price in snap:
        if ts_s < cutoff:
            continue
        cnt += 1
        if side == "LONG":
            long_qty += qty
        else:
            short_qty += qty
    return long_qty, short_qty, cnt


def slope(points):
    if len(points) < 5:
        return 0.0
    t0, v0 = points[0]
    t1, v1 = points[-1]
    dt = max(1e-6, t1 - t0)
    return (v1 - v0) / dt


def oi_slope_for_tf(now_s, window_sec):
    """
    Use up to 1h for OI slope (fast enough), but at least 10m,
    and never more than TF window.
    """
    slope_win = max(10 * 60, min(window_sec, 60 * 60))
    cutoff = now_s - slope_win

    with state_lock:
        pts = [p for p in oi_history if p[0] >= cutoff]
    return slope(pts)


def stress_score(imb, cvd, oi_slope_per_s, funding, basis, long_liq_qty, short_liq_qty, window_sec):
    """
    Normalize by sqrt(window) to keep thresholds meaningful across TFs.
    """
    import math
    norm = max(1.0, math.sqrt(window_sec / (15 * 60)))  # baseline 15m

    score = 0.0

    # Flow
    if cvd < 0:
        score += min(3.0, abs(cvd) / (250.0 * norm))
    if imb < -0.15:
        score += min(2.0, abs(imb) * 5)

    # Leverage regime
    if oi_slope_per_s < 0 and cvd < 0:
        score += 2.5  # unwind + selling
    if oi_slope_per_s > 0 and cvd < 0:
        score += 2.0  # leverage enters into sell flow

    # Funding crowding
    if funding > 0.0005:
        score += 1.5
    elif funding < -0.0005:
        score += 1.0

    # Basis
    if basis < -0.001:
        score += 1.5
    elif basis > 0.001:
        score += 1.0

    # Liquidations amplifier
    liq_total = long_liq_qty + short_liq_qty
    score += min(2.5, liq_total / (800.0 * norm))

    return score


def classify_mode(cvd, oi_slope_per_s, long_liq_qty, short_liq_qty, funding):
    # liquidation dominance
    if long_liq_qty > short_liq_qty * 1.5 and long_liq_qty > 200:
        return "LONG_SQUEEZE"
    if short_liq_qty > long_liq_qty * 1.5 and short_liq_qty > 200:
        return "SHORT_SQUEEZE"

    # fallback
    if cvd < 0 and oi_slope_per_s < 0:
        return "LONG_SQUEEZE"
    if cvd > 0 and oi_slope_per_s < 0 and funding < 0:
        return "SHORT_SQUEEZE"
    return "VOLATILE"


# =========================
# WINDOWS ALARM + POPUP
# =========================
def alarm_loop():
    global alarm_active
    while alarm_active:
        winsound.Beep(2000, 800)
        time.sleep(0.1)


def start_alarm():
    global alarm_active
    if not alarm_active:
        alarm_active = True
        threading.Thread(target=alarm_loop, daemon=True).start()
        print("🔊 SIRENA ATTIVA (tacitala dal popup)")


def stop_alarm():
    global alarm_active
    if alarm_active:
        alarm_active = False
        print("🔇 Sirena silenziata")


def snooze(minutes=5):
    global snooze_until
    snooze_until = time.time() + minutes * 60
    stop_alarm()
    print(f"⏸ Snooze {minutes} minuti")


def copy_to_clipboard(text: str):
    r = tk.Tk()
    r.withdraw()
    r.clipboard_clear()
    r.clipboard_append(text)
    r.update()
    r.destroy()


def build_plan(mode, triggered_tf, display_tf, mid, imb, cvd, oi_slope, funding, basis,
               long_liq_qty, short_liq_qty, liq_cnt, score, global_tf):
    if mode == "LONG_SQUEEZE":
        header = f"🟥 LONG SQUEEZE ({triggered_tf}) — rischio dump rapido"
    elif mode == "SHORT_SQUEEZE":
        header = f"🟩 SHORT SQUEEZE ({triggered_tf}) — rischio spike up"
    else:
        header = f"⚠️ VOLATILE ({triggered_tf}) — rischio move violento"

    meta = [
        f"SYMBOL: {SYMBOL_UP} | DISPLAY={display_tf} | TRIGGER={triggered_tf}",
        f"GLOBAL RED TF: {global_tf if global_tf else 'none'}",
        f"Mid: {mid:.2f}",
        f"Imbalance(book): {imb:+.3f} (neg=ask domina)",
        f"CVD({triggered_tf}): {cvd:+.1f} (neg=taker sell)",
        f"OI slope: {oi_slope:+.2f}/s (+leva entra, -leva esce)",
        f"Funding: {funding:+.6f}",
        f"Basis(mark-index): {basis:+.5f}",
        f"Liquidations({triggered_tf}): LONG={long_liq_qty:.2f} SHORT={short_liq_qty:.2f} (events={liq_cnt})",
        f"Stress score: {score:.2f} | RED>{TF_THRESHOLDS[triggered_tf]['RED']} CLEAR<{TF_THRESHOLDS[triggered_tf]['CLEAR']}",
        "",
    ]

    if mode == "LONG_SQUEEZE":
        actions = [
            "✅ COSA FARE (LONG SQUEEZE):",
            "1) Se sei LONG: riduci subito, SL stretto, NON mediare.",
            "2) Evita catch-the-knife: attendi breakdown+retest o reclaim chiaro.",
            "3) SHORT solo su conferma (retest fallito), size piccola.",
            "4) Attenzione a wick: niente entry a mercato senza trigger.",
        ]
    elif mode == "SHORT_SQUEEZE":
        actions = [
            "✅ COSA FARE (SHORT SQUEEZE):",
            "1) Se sei SHORT: proteggi, SL, non aggiungere impulsivamente.",
            "2) LONG solo su reclaim+hold (breakout + conferma).",
            "3) Prendi profitto a step: spike+retrace è comune.",
            "4) Se funding negativo: squeeze può accelerare.",
        ]
    else:
        actions = [
            "✅ COSA FARE (VOLATILE):",
            "1) Riduci leva e size: priorità difesa.",
            "2) Aspetta conferma (breakdown/retest o reclaim/hold).",
            "3) Evita chop: non entrare in mezzo al range.",
            "4) Se la sirena suona: NON aprire nuove posizioni “di riflesso”.",
        ]

    tail = [
        "",
        "🛑 Sirena: Silenzia manualmente o Snooze 5m.",
        "📋 'Copia piano' copia tutto negli appunti.",
    ]

    return header, "\n".join(meta + actions + tail)


def show_popup(header, plan_text, popup_cooldown_sec):
    global popup_active, last_popup_ts

    now = time.time()
    if popup_active:
        return
    if now - last_popup_ts < popup_cooldown_sec:
        return

    popup_active = True
    last_popup_ts = now

    def _run():
        global popup_active

        root = tk.Tk()
        root.title("ETH STRIKE ALERT")
        root.attributes("-topmost", True)
        root.geometry("650x380")
        root.resizable(False, False)

        frm = ttk.Frame(root, padding=12)
        frm.pack(fill="both", expand=True)

        title = ttk.Label(frm, text=header, font=("Segoe UI", 12, "bold"))
        title.pack(anchor="w")

        txt = tk.Text(frm, height=13, wrap="word", font=("Consolas", 10))
        txt.pack(fill="both", expand=True, pady=(10, 10))
        txt.insert("1.0", plan_text)
        txt.config(state="disabled")

        btns = ttk.Frame(frm)
        btns.pack(fill="x")

        def on_silence():
            stop_alarm()

        def on_snooze():
            snooze(5)
            root.destroy()

        def on_copy():
            copy_to_clipboard(plan_text)

        def on_close():
            root.destroy()

        ttk.Button(btns, text="🔇 Silenzia (sirena)", command=on_silence).pack(side="left", padx=(0, 8))
        ttk.Button(btns, text="⏸ Snooze 5m", command=on_snooze).pack(side="left", padx=(0, 8))
        ttk.Button(btns, text="📋 Copia piano", command=on_copy).pack(side="left", padx=(0, 8))
        ttk.Button(btns, text="Chiudi popup", command=on_close).pack(side="right")

        root.protocol("WM_DELETE_WINDOW", on_close)
        root.mainloop()
        popup_active = False

    threading.Thread(target=_run, daemon=True).start()


# =========================
# ASYNC LOOPS
# =========================
async def rest_poll_loop():
    fail_streak = 0

    while True:
        try:
            now_s = time.time()
            oi = get_open_interest()
            funding, mark, index = get_funding_mark_index()

            with state_lock:
                oi_history.append((now_s, oi))
                funding_history.append((now_s, funding, mark, index))

            fail_streak = 0
            await asyncio.sleep(REST_POLL_SEC)

        except Exception as e:
            fail_streak += 1
            wait = min(
                60,
                REST_POLL_SEC * (1 + fail_streak)
            ) + random.uniform(0.5, 1.5)

            print(
                f"[REST] error (streak={fail_streak}): {e} | retry in ~{wait:.1f}s"
            )

            await asyncio.sleep(wait)
    while True:
        try:
            now_s = time.time()
            oi = get_open_interest()
            funding, mark, index = get_funding_mark_index()
            with state_lock:
                oi_history.append((now_s, oi))
                funding_history.append((now_s, funding, mark, index))
        except Exception as e:
            print(f"[REST] error: {e}")
        await asyncio.sleep(REST_POLL_SEC)


async def ws_loop(display_tf, profile_cfg):
    global last_mid, bids_cache, asks_cache
    global GLOBAL_RED_TF_RANK, GLOBAL_RED_TF_LABEL
    global last_alert_mode

    popup_cooldown = profile_cfg["popup_cooldown"]
    auto_clear = profile_cfg["auto_clear"]
    force_popup_on_global_red = profile_cfg["force_popup_on_global_red"]

    async with websockets.connect(STREAM, ping_interval=20, ping_timeout=20) as ws:
        print(f"\n✅ Connected: {SYMBOL_UP} (aggTrade + depth + liquidations)")
        print(f"📺 DISPLAY TF: {display_tf} | 🎚 PROFILE: {PROFILE.upper()}")
        print("🧠 Calcolo attivo su: 5m/15m/30m/1h/2h/4h (auto-switch gerarchico)\n")

        last_print = 0.0

        while True:
            raw = await ws.recv()
            msg = json.loads(raw)

            stream = msg.get("stream", "")
            data = msg.get("data", {})

            now_s = time.time()
            prune_ticks(now_s)

            if stream.endswith("@aggTrade"):
                ts = int(data["T"])
                qty = float(data["q"])
                price = float(data["p"])
                is_buyer_maker = bool(data["m"])
                with state_lock:
                    trades.append((ts, qty, is_buyer_maker, price))

            elif "@depth" in stream:
                bids = data.get("b", [])
                asks = data.get("a", [])
                with state_lock:
                    bids_cache = bids[:100]
                    asks_cache = asks[:100]
                mid = compute_mid(bids_cache, asks_cache)
                if mid is not None:
                    last_mid = mid

            elif stream.endswith("@forceOrder"):
                o = data.get("o", {})
                side = o.get("S", "")  # "SELL" liquidates LONGs; "BUY" liquidates SHORTs
                qty = float(o.get("q", 0))
                price = float(o.get("p", 0))
                liq_side = "LONG" if side == "SELL" else "SHORT"
                with state_lock:
                    liq_events.append((now_s, liq_side, qty, price))

            # Need baseline data
            if last_mid and len(funding_history) >= 1 and len(oi_history) >= 5:
                now_ms = int(now_s * 1000)

                with state_lock:
                    bids = list(bids_cache)
                    asks = list(asks_cache)
                    funding, mark, index = funding_history[-1][1], funding_history[-1][2], funding_history[-1][3]
                    snooze_local = snooze_until

                imb = depth_imbalance(bids, asks)
                basis = (mark - index) / index if index else 0.0

                # Compute per TF
                tf_results = {}
                highest_red_rank = 0
                highest_red_tf = None

                for tf_label, window_sec in TF_OPTIONS.items():
                    cvd, buy, sell = cvd_window(now_ms, window_sec)
                    oi_sl = oi_slope_for_tf(now_s, window_sec)
                    long_liq, short_liq, liq_cnt = liq_stats(now_s, window_sec)

                    score = stress_score(imb, cvd, oi_sl, funding, basis, long_liq, short_liq, window_sec)
                    mode = classify_mode(cvd, oi_sl, long_liq, short_liq, funding)

                    red_th = TF_THRESHOLDS[tf_label]["RED"]
                    clear_th = TF_THRESHOLDS[tf_label]["CLEAR"]

                    is_red = score >= red_th
                    is_clear = score < clear_th

                    tf_results[tf_label] = {
                        "window": window_sec,
                        "cvd": cvd,
                        "oi_slope": oi_sl,
                        "liq_long": long_liq,
                        "liq_short": short_liq,
                        "liq_cnt": liq_cnt,
                        "score": score,
                        "mode": mode,
                        "is_red": is_red,
                        "is_clear": is_clear,
                        "red_th": red_th,
                        "clear_th": clear_th,
                    }

                    if is_red and TF_RANK[tf_label] > highest_red_rank:
                        highest_red_rank = TF_RANK[tf_label]
                        highest_red_tf = tf_label

                # Update global RED TF
                if highest_red_tf is not None:
                    GLOBAL_RED_TF_RANK = highest_red_rank
                    GLOBAL_RED_TF_LABEL = highest_red_tf
                else:
                    # If none red, clear global
                    GLOBAL_RED_TF_RANK = 0
                    GLOBAL_RED_TF_LABEL = None

                # Print DISPLAY TF once per second
                if now_s - last_print >= PRINT_EVERY_SEC:
                    last_print = now_s
                    d = tf_results[display_tf]
                    print(
                        f"[{display_tf}] mid={last_mid:.2f} imb={imb:+.3f} "
                        f"cvd={d['cvd']:+.1f} OI_slope={d['oi_slope']:+.2f}/s "
                        f"fund={funding:+.6f} basis={basis:+.5f} "
                        f"liq L={d['liq_long']:.1f} S={d['liq_short']:.1f} cnt={d['liq_cnt']} "
                        f"score={d['score']:.2f} (RED>{d['red_th']}) "
                        f"| GLOBAL_RED={GLOBAL_RED_TF_LABEL or 'none'}"
                    )

                # ALERT LOGIC (auto-switch)
                if now_s >= snooze_local:
                    # Determine effective RED:
                    # - if display tf is red
                    # - OR if a higher TF is red and profile allows forcing
                    display_is_red = tf_results[display_tf]["is_red"]
                    higher_red_exists = (GLOBAL_RED_TF_RANK > TF_RANK[display_tf])

                    effective_red = display_is_red or (force_popup_on_global_red and higher_red_exists)

                    if effective_red:
                        start_alarm()

                        # Trigger TF = the highest red (more systemic) if exists, else display TF
                        trigger_tf = GLOBAL_RED_TF_LABEL or display_tf
                        t = tf_results[trigger_tf]

                        header, plan_text = build_plan(
                            mode=t["mode"],
                            triggered_tf=trigger_tf,
                            display_tf=display_tf,
                            mid=last_mid,
                            imb=imb,
                            cvd=t["cvd"],
                            oi_slope=t["oi_slope"],
                            funding=funding,
                            basis=basis,
                            long_liq_qty=t["liq_long"],
                            short_liq_qty=t["liq_short"],
                            liq_cnt=t["liq_cnt"],
                            score=t["score"],
                            global_tf=GLOBAL_RED_TF_LABEL,
                        )

                        # Popup rules:
                        # - show if mode changed OR cooldown allows
                        if t["mode"] != last_alert_mode:
                            last_alert_mode = t["mode"]
                            show_popup(header, plan_text, popup_cooldown)
                        else:
                            show_popup(header, plan_text, popup_cooldown)

                    # Auto clear only if profile allows AND no global red
                    if auto_clear:
                        if GLOBAL_RED_TF_LABEL is None:
                            # optional: also require display tf below CLEAR
                            if tf_results[display_tf]["is_clear"]:
                                stop_alarm()
                                last_alert_mode = None

            await asyncio.sleep(0.02)


async def main(display_tf, profile_cfg):
    await asyncio.gather(
        rest_poll_loop(),
        ws_loop(display_tf, profile_cfg),
    )


# =========================
# ENTRYPOINT
# =========================
if __name__ == "__main__":
    DISPLAY_TF = choose_display_timeframe()
    PROFILE = choose_profile()
    PROFILE_CFG = PROFILES[PROFILE]

    try:
        asyncio.run(main(DISPLAY_TF, PROFILE_CFG))
    except KeyboardInterrupt:
        print("Stopped.")
