
from __future__ import annotations
"""
MM Push Monitor — ETH (Binance Futures) | Dashboard ITA + Alert sonoro continuo (Windows)

Segnale "momento esatto" (spesso pochi minuti):
- Breakout dal range recente + spike di volume + espansione volatilità (Bollinger bandwidth)
- Conferma microstrutturale: imbalance orderbook (bid/ask) a favore del verso
- (Opzionale) Δ Open Interest via REST (polling) per capire se è short-covering o nuova leva

ATTENZIONE: è un tool informativo, non consulenza finanziaria.

Avvio:
    python mm_push_monitor_it.py
Install:
    pip install -r requirements.txt
"""
import json
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple
from collections import deque

import requests
import websocket  # websocket-client

import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:
    from plyer import notification
except Exception:
    notification = None

try:
    import winsound  # Windows
except Exception:
    winsound = None


# ---------- CONFIG ----------
REST = "https://fapi.binance.com"
WS = "wss://fstream.binance.com/ws"

SYMBOL = "ETHUSDT"
STREAM_MARK = "ethusdt@markPrice@1s"
STREAM_DEPTH = "ethusdt@depth20@100ms"
STREAM_K5 = "ethusdt@kline_5m"
STREAM_K15 = "ethusdt@kline_15m"

OI_POLL_SEC = 10
UI_REFRESH_MS = 250

# indicatori
RANGE_LOOKBACK = 20
BB_PERIOD = 20
BB_STD = 2.0
RSI_PERIOD = 14
VOL_Z_LOOKBACK = 20
SQUEEZE_LOOKBACK = 12

# soglie segnale
BREAK_PCT = 0.0010      # 0.10%
IMB_TH = 0.12
VOL_Z_TH = 1.5
BAND_EXPAND_TH = 1.15

# suono continuo
BEEP_FREQ = 880
BEEP_DUR_MS = 350
BEEP_PAUSE_MS = 150


# ---------- UTILS ----------
def now() -> float:
    return time.time()

def fmt(x: float, d: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:,.{d}f}"

def notify(title: str, msg: str):
    if notification is None:
        return
    try:
        notification.notify(title=title, message=msg, timeout=6)
    except Exception:
        pass

def sma(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")

def stdev(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = sma(vals)
    v = sum((x - m) ** 2 for x in vals) / len(vals)
    return math.sqrt(v)

def bollinger(close: List[float], period: int = 20, std_mult: float = 2.0):
    if len(close) < period:
        return float("nan"), float("nan"), float("nan"), float("nan")
    w = close[-period:]
    m = sma(w)
    s = stdev(w)
    up = m + std_mult * s
    dn = m - std_mult * s
    bw = (up - dn) / m if m != 0 else float("nan")
    return m, up, dn, bw

def rsi(close: List[float], period: int = 14) -> float:
    if len(close) < period + 1:
        return float("nan")
    w = close[-(period + 1):]
    g = 0.0
    l = 0.0
    for i in range(1, len(w)):
        d = w[i] - w[i - 1]
        if d > 0:
            g += d
        else:
            l += -d
    if l == 0 and g == 0:
        return 50.0
    if l == 0:
        return 100.0
    rs = g / l
    return 100.0 - 100.0 / (1 + rs)

def zscore(vals: List[float]) -> float:
    if len(vals) < 5:
        return float("nan")
    m = sma(vals)
    s = stdev(vals)
    if s == 0:
        return 0.0
    return (vals[-1] - m) / s

def orderbook_imbalance(bids: List[Tuple[float, float]], asks: List[Tuple[float, float]], mid: float, window_pct: float = 0.01) -> float:
    if not bids or not asks or not mid or math.isnan(mid):
        return float("nan")
    lo = mid * (1 - window_pct)
    hi = mid * (1 + window_pct)
    bn = 0.0
    an = 0.0
    for p, q in bids:
        if lo <= p <= hi:
            bn += p * q
    for p, q in asks:
        if lo <= p <= hi:
            an += p * q
    denom = bn + an
    if denom <= 0:
        return float("nan")
    return (bn - an) / denom


# ---------- DATA ----------
@dataclass
class Candle:
    t_close: int
    o: float
    h: float
    l: float
    c: float
    v: float
    qv: float

@dataclass
class TFState:
    tf_label: str
    candles: Deque[Candle] = field(default_factory=lambda: deque(maxlen=500))
    last_signal: str = "—"
    last_reason: str = "—"
    last_ts: float = 0.0

@dataclass
class Snapshot:
    ts: float = 0.0
    mark: float = float("nan")
    bids: List[Tuple[float, float]] = field(default_factory=list)
    asks: List[Tuple[float, float]] = field(default_factory=list)
    imb: float = float("nan")
    oi: float = float("nan")
    oi_prev: float = float("nan")
    oi_ts: float = 0.0


# ---------- ENGINE ----------
class Engine:
    def __init__(self):
        self.q: "queue.Queue[dict]" = queue.Queue(maxsize=4000)
        self.stop_ev = threading.Event()
        self.lock = threading.Lock()
        self.snap = Snapshot()
        self.tf5 = TFState("5m")
        self.tf15 = TFState("15m")
        self._ws = []

    def start(self):
        for stream, tag in [(STREAM_MARK, "mark"), (STREAM_DEPTH, "depth"), (STREAM_K5, "k5"), (STREAM_K15, "k15")]:
            t = threading.Thread(target=self._ws_loop, args=(stream, tag), daemon=True)
            t.start()
        threading.Thread(target=self._oi_poll, daemon=True).start()

    def stop(self):
        self.stop_ev.set()
        for ws in self._ws:
            try:
                ws.close()
            except Exception:
                pass

    def _ws_loop(self, stream: str, tag: str):
        def on_message(ws, msg):
            try:
                self.q.put_nowait({"tag": tag, "data": json.loads(msg), "ts": now()})
            except Exception:
                pass

        while not self.stop_ev.is_set():
            try:
                wsa = websocket.WebSocketApp(f"{WS}/{stream}", on_message=on_message)
                self._ws.append(wsa)
                wsa.run_forever(ping_interval=20, ping_timeout=10)
            except Exception:
                pass
            self.stop_ev.wait(2.0)

    def _oi_poll(self):
        while not self.stop_ev.is_set():
            try:
                r = requests.get(f"{REST}/fapi/v1/openInterest", params={"symbol": SYMBOL}, timeout=8)
                r.raise_for_status()
                oi = float(r.json()["openInterest"])
                with self.lock:
                    self.snap.oi_prev = self.snap.oi
                    self.snap.oi = oi
                    self.snap.oi_ts = now()
            except Exception:
                pass
            self.stop_ev.wait(OI_POLL_SEC)

    def pump(self):
        while True:
            try:
                item = self.q.get_nowait()
            except queue.Empty:
                break
            tag = item["tag"]
            data = item["data"]
            ts = item["ts"]

            if tag == "mark":
                with self.lock:
                    self.snap.ts = ts
                    self.snap.mark = float(data.get("p", self.snap.mark))

            elif tag == "depth":
                bids = [(float(p), float(q)) for p, q in data.get("b", [])]
                asks = [(float(p), float(q)) for p, q in data.get("a", [])]
                bids.sort(key=lambda x: x[0], reverse=True)
                asks.sort(key=lambda x: x[0])
                with self.lock:
                    self.snap.ts = ts
                    self.snap.bids = bids
                    self.snap.asks = asks
                    mid = self.snap.mark if not math.isnan(self.snap.mark) else ((bids[0][0] + asks[0][0]) / 2 if bids and asks else float("nan"))
                    self.snap.imb = orderbook_imbalance(bids, asks, mid, window_pct=0.01)

            elif tag in ("k5", "k15"):
                k = data.get("k", {})
                if not k or not bool(k.get("x", False)):
                    continue  # solo chiusure
                cndl = Candle(
                    t_close=int(k.get("T", 0)),
                    o=float(k.get("o", 0)),
                    h=float(k.get("h", 0)),
                    l=float(k.get("l", 0)),
                    c=float(k.get("c", 0)),
                    v=float(k.get("v", 0)),
                    qv=float(k.get("q", 0)),
                )
                st = self.tf5 if tag == "k5" else self.tf15
                st.candles.append(cndl)
                self._compute_signal(st)

    def _compute_signal(self, st: TFState):
        closes = [c.c for c in st.candles]
        highs = [c.h for c in st.candles]
        lows = [c.l for c in st.candles]
        vols = [c.qv for c in st.candles]  # quote volume

        need = max(RANGE_LOOKBACK + 2, BB_PERIOD + 2, RSI_PERIOD + 2, VOL_Z_LOOKBACK + 2)
        if len(closes) < need:
            return

        range_hi = max(highs[-(RANGE_LOOKBACK + 1):-1])
        range_lo = min(lows[-(RANGE_LOOKBACK + 1):-1])
        c = closes[-1]

        mid, up, dn, bw = bollinger(closes, BB_PERIOD, BB_STD)

        # bandwidth history for squeeze/expand
        bws = []
        for i in range(BB_PERIOD, len(closes) + 1):
            _, _, _, bw_i = bollinger(closes[:i], BB_PERIOD, BB_STD)
            if not math.isnan(bw_i):
                bws.append(bw_i)
        bw_recent = bws[-SQUEEZE_LOOKBACK:] if len(bws) >= SQUEEZE_LOOKBACK else bws
        bw_avg = sma(bw_recent) if bw_recent else float("nan")
        bw_min = min(bw_recent) if bw_recent else float("nan")
        squeeze = (not math.isnan(bw_min) and not math.isnan(bw) and bw <= bw_min * 1.20)
        expand = (not math.isnan(bw_avg) and not math.isnan(bw) and bw >= bw_avg * BAND_EXPAND_TH)

        r = rsi(closes, RSI_PERIOD)

        vz = zscore(vols[-VOL_Z_LOOKBACK:])

        with self.lock:
            imb = self.snap.imb
            oi = self.snap.oi
            oi_prev = self.snap.oi_prev
        oi_delta = float("nan")
        if not math.isnan(oi) and not math.isnan(oi_prev):
            oi_delta = oi - oi_prev

        bull_break = c >= range_hi * (1 + BREAK_PCT)
        bear_break = c <= range_lo * (1 - BREAK_PCT)
        vol_ok = (not math.isnan(vz) and vz >= VOL_Z_TH)
        imb_bull = (not math.isnan(imb) and imb >= IMB_TH)
        imb_bear = (not math.isnan(imb) and imb <= -IMB_TH)
        bull_bb = (not math.isnan(up) and c >= up) or (not math.isnan(mid) and c >= mid)
        bear_bb = (not math.isnan(dn) and c <= dn) or (not math.isnan(mid) and c <= mid)

        signal = "—"
        reasons = []

        if bull_break and vol_ok and (expand or squeeze) and imb_bull and bull_bb and (math.isnan(r) or r >= 48):
            signal = "MM PUSH ↑ (spinta rialzista)"
            reasons = [
                "Breakout sopra range recente",
                f"Volume spike (z={vz:.2f})",
                f"Volatilità in espansione (BW={bw:.4f})",
                f"Imbalance pro-bid ({imb:+.2f})",
            ]
            if not math.isnan(oi_delta):
                reasons.append(f"ΔOI={oi_delta:+.0f} ETH (copertura o nuova leva)")

        elif bear_break and vol_ok and (expand or squeeze) and imb_bear and bear_bb and (math.isnan(r) or r <= 52):
            signal = "MM PUSH ↓ (spinta ribassista)"
            reasons = [
                "Breakdown sotto range recente",
                f"Volume spike (z={vz:.2f})",
                f"Volatilità in espansione (BW={bw:.4f})",
                f"Imbalance pro-ask ({imb:+.2f})",
            ]
            if not math.isnan(oi_delta):
                reasons.append(f"ΔOI={oi_delta:+.0f} ETH (copertura o nuova leva)")

        else:
            if not math.isnan(bw) and not math.isnan(bw_avg) and bw < bw_avg and (math.isnan(vz) or vz < 0.5):
                signal = "Accumulo/Compressione (attesa)"
                reasons = [
                    "Bande strette / volatilità compressa",
                    "Volume non esplosivo",
                    "Aspetta breakout con conferme",
                ]

        if signal != st.last_signal:
            st.last_signal = signal
            st.last_reason = " | ".join(reasons) if reasons else "—"
            st.last_ts = now()


# ---------- ALERT ----------
class AlertController:
    def __init__(self):
        self._active = threading.Event()
        self._stop = threading.Event()

    def start(self, title: str, message: str, enable_sound: bool = True):
        if self._active.is_set():
            return
        self._active.set()
        self._stop.clear()

        notify(title, message)

        if enable_sound:
            threading.Thread(target=self._sound_loop, daemon=True).start()

    def stop(self):
        self._stop.set()
        self._active.clear()

    def _sound_loop(self):
        while not self._stop.is_set():
            if winsound is not None:
                try:
                    winsound.Beep(BEEP_FREQ, BEEP_DUR_MS)
                except Exception:
                    pass
            else:
                print("\a", end="", flush=True)
            time.sleep(BEEP_PAUSE_MS / 1000.0)


# ---------- GUI ----------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MM Push Monitor ETH — 5m & 15m (Dashboard + Alert continuo)")
        self.geometry("1250x820")

        self.engine = Engine()
        self.alert = AlertController()
        self.var_sound = tk.BooleanVar(value=True)

        self._popup: Optional[tk.Toplevel] = None
        self._last_alert_key = ""

        self._build_ui()
        self.engine.start()
        self.after(UI_REFRESH_MS, self._tick)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        try:
            self.engine.stop()
        except Exception:
            pass
        self.alert.stop()
        self.destroy()

    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        self.lbl_price = ttk.Label(top, text="Mark: —", font=("Segoe UI", 16, "bold"))
        self.lbl_price.pack(side=tk.LEFT, padx=(0, 14))

        self.lbl_imb = ttk.Label(top, text="Imbalance: —", font=("Segoe UI", 12))
        self.lbl_imb.pack(side=tk.LEFT, padx=(0, 14))

        self.lbl_oi = ttk.Label(top, text="Open Interest: —", font=("Segoe UI", 12))
        self.lbl_oi.pack(side=tk.LEFT, padx=(0, 14))

        ttk.Checkbutton(top, text="Suono continuo", variable=self.var_sound).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(top, text="FERMA ALLARME", command=self._stop_alarm).pack(side=tk.RIGHT)

        sig = ttk.Labelframe(self, text="Segnali (solo a chiusura candela)", padding=10)
        sig.pack(side=tk.TOP, fill=tk.X, padx=10)

        self.lbl_s5 = ttk.Label(sig, text="5m: —", font=("Segoe UI", 13, "bold"))
        self.lbl_s5.grid(row=0, column=0, sticky="w", padx=(0, 14))

        self.lbl_s15 = ttk.Label(sig, text="15m: —", font=("Segoe UI", 13, "bold"))
        self.lbl_s15.grid(row=1, column=0, sticky="w", padx=(0, 14), pady=(6, 0))

        self.lbl_r5 = ttk.Label(sig, text="Motivo 5m: —", font=("Segoe UI", 10), wraplength=1050, justify="left")
        self.lbl_r5.grid(row=0, column=1, sticky="w")

        self.lbl_r15 = ttk.Label(sig, text="Motivo 15m: —", font=("Segoe UI", 10), wraplength=1050, justify="left")
        self.lbl_r15.grid(row=1, column=1, sticky="w", pady=(6, 0))

        body = ttk.Frame(self, padding=10)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        fig = Figure(figsize=(8.2, 6.2), dpi=100)
        self.ax5 = fig.add_subplot(211)
        self.ax15 = fig.add_subplot(212)
        fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(fig, master=body)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = ttk.Frame(body)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        logbox = ttk.Labelframe(right, text="Log segnali", padding=10)
        logbox.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.txt = tk.Text(logbox, width=44, height=26)
        self.txt.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _tick(self):
        self.engine.pump()
        self._update_ui()
        self.after(UI_REFRESH_MS, self._tick)

    def _stop_alarm(self):
        self.alert.stop()
        if self._popup is not None:
            try:
                self._popup.destroy()
            except Exception:
                pass
            self._popup = None

    def _ensure_popup(self, title: str, message: str):
        if self._popup is not None:
            return
        pop = tk.Toplevel(self)
        pop.title(title)
        pop.attributes("-topmost", True)
        pop.geometry("520x220+80+80")
        ttk.Label(pop, text=title, font=("Segoe UI", 14, "bold")).pack(pady=(12, 6))
        ttk.Label(pop, text=message, wraplength=480, justify="left").pack(pady=(0, 10))
        ttk.Button(pop, text="FERMA ALLARME", command=self._stop_alarm).pack(pady=10)
        self._popup = pop

    def _update_ui(self):
        with self.engine.lock:
            mark = self.engine.snap.mark
            imb = self.engine.snap.imb
            oi = self.engine.snap.oi
            oi_prev = self.engine.snap.oi_prev

        self.lbl_price.config(text=f"Mark: {fmt(mark, 2)}")
        self.lbl_imb.config(text=f"Imbalance: {('—' if math.isnan(imb) else f'{imb:+.2f}')}")
        if not math.isnan(oi):
            d = "" if math.isnan(oi_prev) else f" (Δ {oi - oi_prev:+.0f} ETH)"
            self.lbl_oi.config(text=f"Open Interest: {fmt(oi, 0)} ETH{d}")
        else:
            self.lbl_oi.config(text="Open Interest: —")

        s5, r5, ts5 = self.engine.tf5.last_signal, self.engine.tf5.last_reason, self.engine.tf5.last_ts
        s15, r15, ts15 = self.engine.tf15.last_signal, self.engine.tf15.last_reason, self.engine.tf15.last_ts

        self.lbl_s5.config(text=f"5m: {s5}")
        self.lbl_s15.config(text=f"15m: {s15}")
        self.lbl_r5.config(text=f"Motivo 5m: {r5}")
        self.lbl_r15.config(text=f"Motivo 15m: {r15}")

        self._draw_charts()
        self._maybe_alert("5m", ts5, s5, r5)
        self._maybe_alert("15m", ts15, s15, r15)

    def _draw_charts(self):
        c5 = list(self.engine.tf5.candles)
        c15 = list(self.engine.tf15.candles)

        if c5:
            closes = [c.c for c in c5]
            self.ax5.clear()
            self.ax5.set_title("Chiusure 5m + Bollinger")
            self.ax5.plot(closes, label="Close")
            mids, ups, dns = [], [], []
            for i in range(len(closes)):
                m, u, d, _ = bollinger(closes[:i + 1], BB_PERIOD, BB_STD)
                mids.append(m); ups.append(u); dns.append(d)
            self.ax5.plot(mids, label="MB")
            self.ax5.plot(ups, label="UP")
            self.ax5.plot(dns, label="DN")
            self.ax5.grid(True, alpha=0.25)
            self.ax5.legend(loc="upper left", fontsize=8)

        if c15:
            closes = [c.c for c in c15]
            self.ax15.clear()
            self.ax15.set_title("Chiusure 15m + Bollinger")
            self.ax15.plot(closes, label="Close")
            mids, ups, dns = [], [], []
            for i in range(len(closes)):
                m, u, d, _ = bollinger(closes[:i + 1], BB_PERIOD, BB_STD)
                mids.append(m); ups.append(u); dns.append(d)
            self.ax15.plot(mids, label="MB")
            self.ax15.plot(ups, label="UP")
            self.ax15.plot(dns, label="DN")
            self.ax15.grid(True, alpha=0.25)
            self.ax15.legend(loc="upper left", fontsize=8)

        self.canvas.draw_idle()

    def _maybe_alert(self, tf: str, last_ts: float, signal: str, reason: str):
        if not signal.startswith("MM PUSH"):
            return
        key = f"{tf}|{signal}|{int(last_ts)}"
        if key == self._last_alert_key:
            return
        self._last_alert_key = key

        title = f"ALLARME {tf}: {signal}"
        msg = reason if reason != "—" else signal

        self.txt.insert("end", f"[{time.strftime('%H:%M:%S')}] {title} -> {msg}\n")
        self.txt.see("end")

        self.alert.start(title, msg, enable_sound=self.var_sound.get())
        self._ensure_popup(title, msg)


def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()
