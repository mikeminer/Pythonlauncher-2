
"""
EMDC - "Mamma Mode" (versione super semplice) - v1.0
----------------------------------------------------
Obiettivo: far inserire pochissimi dati essenziali e mostrare subito:
- piano martingala esponenziale (cap + assorbimento)
- prezzi di ricarico auto-calcolati da dati live Binance (REST)
- mini report indicatori (RSI/MACD/ADX/BB)
- grafico Prezzo vs Curva

IMPORTANTE (leggere):
- Questo NON e' un modo garantito per "fare soldi".
- Trading con leva e martingala puo' portare a perdite molto rapide.
- Usa sempre importi piccoli, preferibilmente in demo/paper.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import math
import time

try:
    import requests
except Exception:
    requests = None

# plotting
try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except Exception:
    Figure = None
    FigureCanvasTkAgg = None

BINANCE_BASE = "https://api.binance.com"

@dataclass
class PlanRow:
    step: int
    step_size: float
    total_size: float
    reload_price: Optional[float]
    note: str

@dataclass
class LiveSnapshot:
    symbol: str
    interval: str
    ts: float
    price: float
    volume: float
    avg_volatility_pct: float
    rsi14: float
    macd_mode: str
    adx_mode: str
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_note: str

def f2(x: float) -> str:
    return f"{x:.6f}".rstrip("0").rstrip(".")

def parse_float(s: str, field: str) -> float:
    try:
        v = float(str(s).replace(",", ".").strip())
    except Exception:
        raise ValueError(f"Valore non valido per '{field}'.")
    if v < 0:
        raise ValueError(f"'{field}' non puo essere negativo.")
    return v

def parse_int(s: str, field: str) -> int:
    try:
        v = int(str(s).strip())
    except Exception:
        raise ValueError(f"Valore non valido per '{field}'.")
    if v < 0:
        raise ValueError(f"'{field}' non puo essere negativo.")
    return v

# ---------- Indicators (minimal set) ----------
def sma(values: List[float], period: int) -> List[float]:
    out, s = [], 0.0
    for i, v in enumerate(values):
        s += v
        if i >= period:
            s -= values[i - period]
        out.append(s / period if i >= period - 1 else float("nan"))
    return out

def stddev(values: List[float], period: int) -> List[float]:
    out = []
    for i in range(len(values)):
        if i < period - 1:
            out.append(float("nan"))
        else:
            w = values[i - period + 1:i + 1]
            m = sum(w) / period
            out.append(math.sqrt(sum((x - m) ** 2 for x in w) / period))
    return out

def bollinger(closes: List[float], period: int = 20, mult: float = 2.0):
    mid = sma(closes, period)
    sd = stddev(closes, period)
    up, lo = [], []
    for m, s in zip(mid, sd):
        if math.isnan(m) or math.isnan(s):
            up.append(float("nan")); lo.append(float("nan"))
        else:
            up.append(m + mult * s); lo.append(m - mult * s)
    return up, mid, lo

def rsi(closes: List[float], period: int = 14) -> List[float]:
    if len(closes) < period + 1:
        return [float("nan")] * len(closes)
    gains = [0.0]; losses = [0.0]
    for i in range(1, len(closes)):
        d = closes[i] - closes[i-1]
        gains.append(max(d, 0.0)); losses.append(max(-d, 0.0))
    avg_gain = sum(gains[1:period+1]) / period
    avg_loss = sum(losses[1:period+1]) / period
    out = [float("nan")] * period
    out.append(100.0 if avg_loss == 0 else 100.0 - (100.0 / (1.0 + (avg_gain / avg_loss))))
    for i in range(period + 1, len(closes)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        out.append(100.0 if avg_loss == 0 else 100.0 - (100.0 / (1.0 + (avg_gain / avg_loss))))
    return out

def ema(values: List[float], period: int) -> List[float]:
    if len(values) < period:
        return [float("nan")] * len(values)
    k = 2.0 / (period + 1.0)
    out = [float("nan")] * (period - 1)
    prev = sum(values[:period]) / period
    out.append(prev)
    for v in values[period:]:
        prev = v * k + prev * (1 - k)
        out.append(prev)
    return out

def macd_mode(closes: List[float]) -> str:
    fast = ema(closes, 12)
    slow = ema(closes, 26)
    line = []
    for a, b in zip(fast, slow):
        line.append(float("nan") if math.isnan(a) or math.isnan(b) else a - b)
    sig = ema([0.0 if math.isnan(x) else x for x in line], 9)
    if math.isnan(line[-1]) or math.isnan(sig[-1]):
        return "N/A"
    return "Bearish Crossover Mode" if line[-1] < sig[-1] else "Bullish Crossover Mode"

def adx_mode_simple(highs: List[float], lows: List[float], closes: List[float]) -> str:
    # very simplified ADX strength proxy: average range relative to close (not full ADX)
    if len(closes) < 20:
        return "N/A"
    ranges = [(h-l)/c if c else 0 for h,l,c in zip(highs[-20:], lows[-20:], closes[-20:])]
    m = sum(ranges)/len(ranges)
    if m < 0.002:
        return "Weak Trend"
    if m < 0.004:
        return "Moderate Trend"
    return "Strong Trend"

# ---------- Binance ----------
def fetch_klines(symbol: str, interval: str, limit: int = 300):
    if requests is None:
        raise RuntimeError("Manca 'requests'. Installa: pip install requests")
    url = f"{BINANCE_BASE}/api/v3/klines"
    r = requests.get(url, params={"symbol": symbol.upper(), "interval": interval, "limit": limit}, timeout=10)
    r.raise_for_status()
    return r.json()

def fetch_price(symbol: str) -> float:
    if requests is None:
        raise RuntimeError("Manca 'requests'. Installa: pip install requests")
    url = f"{BINANCE_BASE}/api/v3/ticker/price"
    r = requests.get(url, params={"symbol": symbol.upper()}, timeout=10)
    r.raise_for_status()
    return float(r.json()["price"])

def get_live(symbol: str, interval: str) -> LiveSnapshot:
    kl = fetch_klines(symbol, interval, 300)
    o = [float(k[1]) for k in kl]
    h = [float(k[2]) for k in kl]
    l = [float(k[3]) for k in kl]
    c = [float(k[4]) for k in kl]
    v = [float(k[5]) for k in kl]
    price = fetch_price(symbol)
    # volatility avg (last 20)
    volp = []
    for hi, lo, cl in zip(h[-20:], l[-20:], c[-20:]):
        if cl:
            volp.append((hi-lo)/cl)
    avg_vol = (sum(volp)/len(volp))*100 if volp else float("nan")
    rsi14 = rsi(c, 14)[-1]
    mm = macd_mode(c)
    adx_m = adx_mode_simple(h,l,c)
    bbu, bbm, bbl = bollinger(c, 20, 2.0)
    note = "Near Lower Band" if not math.isnan(bbl[-1]) and abs(price-bbl[-1]) < abs(price-bbu[-1]) else "Near Upper Band"
    return LiveSnapshot(symbol=symbol.upper(), interval=interval, ts=time.time(),
                        price=price, volume=v[-1], avg_volatility_pct=avg_vol,
                        rsi14=rsi14, macd_mode=mm, adx_mode=adx_m,
                        bb_upper=bbu[-1], bb_middle=bbm[-1], bb_lower=bbl[-1], bb_note=note)

# ---------- Plan ----------
def compute_plan(S0: float, r: float, max_steps: int, max_total: float, cap_last: bool=True, absorb_last: bool=True) -> List[PlanRow]:
    if S0 <= 0:
        raise ValueError("Importo iniziale deve essere > 0.")
    if r <= 1.0:
        raise ValueError("r deve essere > 1.0")
    rows = []
    used = min(S0, max_total) if cap_last else S0
    if used > max_total:
        raise ValueError("Importo iniziale > MaxTotale. Attiva cap o riduci.")
    rows.append(PlanRow(0, used, used, None, "Entry"))
    for n in range(1, max_steps+1):
        if used >= max_total:
            rows.append(PlanRow(n, 0.0, used, None, "STOP: cap raggiunto"))
            break
        step = S0 * (r ** n)
        if used + step > max_total:
            if cap_last:
                step = max_total - used
                used = max_total
                rows.append(PlanRow(n, step, used, None, "Ultimo step (cap)"))
            else:
                rows.append(PlanRow(n, step, used, None, "NON applicabile (supera cap)"))
            break
        if cap_last and absorb_last and n < max_steps:
            remaining = max_total - (used + step)
            if 0 < remaining < step:
                step = max_total - used
                used = max_total
                rows.append(PlanRow(n, step, used, None, "Ultimo step (assorbito)"))
                break
        used += step
        rows.append(PlanRow(n, step, used, None, "Ricarico"))
    return rows

def auto_prices(rows: List[PlanRow], side: str, live: LiveSnapshot, ldm: float) -> List[PlanRow]:
    """
    Compila i prezzi di ricarico in modo COERENTE con la direzione:
      - LONG  -> prezzi non crescenti (ogni ricarico <= precedente)
      - SHORT -> prezzi non decrescenti (ogni ricarico >= precedente)

    Perche' puo' capitare un "ricarico piu' alto"?
    - Se il prezzo live e' gia' sotto la Lower Band, allora BB Lower risulta sopra al prezzo.
      In quel caso la versione precedente poteva generare 1-2 punti iniziali leggermente piu' alti.
    Qui lo impediamo:
    - clampiamo BB al prezzo (LONG: min(BB_lower, price), SHORT: max(BB_upper, price))
    - imponiamo monotonicita' sulla curva finale.
    """
    anchors = [live.price]

    if side == "LONG":
        bb = live.bb_lower
        if bb is not None and not math.isnan(bb):
            anchors.append(min(bb, live.price))  # clamp
        anchors.append(ldm)
        anchors = [a for a in anchors if a is not None and not math.isnan(a)]
        anchors = [anchors[0]] + sorted(anchors[1:], reverse=True)
        for i in range(1, len(anchors)):
            anchors[i] = min(anchors[i], anchors[i-1])
    else:
        bb = live.bb_upper
        if bb is not None and not math.isnan(bb):
            anchors.append(max(bb, live.price))  # clamp
        anchors.append(ldm)
        anchors = [a for a in anchors if a is not None and not math.isnan(a)]
        anchors = [anchors[0]] + sorted(anchors[1:])
        for i in range(1, len(anchors)):
            anchors[i] = max(anchors[i], anchors[i-1])

    eff = [r for r in rows if r.step_size > 0]
    n = len(eff)
    if n == 0:
        return rows

    ladder = []
    for i in range(n):
        t = 1.0 if n == 1 else i/(n-1)
        t2 = 1.0 - (1.0 - t) ** 2  # densita' verso LDM
        pos = t2 * (len(anchors)-1)
        j = min(int(math.floor(pos)), len(anchors)-2)
        frac = pos - j
        ladder.append(anchors[j] + (anchors[j+1]-anchors[j]) * frac)

    # enforce monotonic ladder
    if side == "LONG":
        for i in range(1, len(ladder)):
            ladder[i] = min(ladder[i], ladder[i-1])
    else:
        for i in range(1, len(ladder)):
            ladder[i] = max(ladder[i], ladder[i-1])

    it = iter(ladder)
    out = []
    for r in rows:
        if r.step_size <= 0:
            out.append(r); continue
        out.append(PlanRow(r.step, r.step_size, r.total_size, next(it), r.note))
    return out

# ---------- GUI ----------
class Wizard(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("EMDC - Avvio rapido")
        self.resizable(False, False)
        self.result = None

        pad = ttk.Frame(self, padding=14)
        pad.pack(fill="both", expand=True)

        msg = ("Inserisci SOLO questi dati essenziali.\n"
               "Consiglio: inizia con importi piccoli.\n"
               "EMDC non garantisce profitti.")
        ttk.Label(pad, text=msg, justify="left").grid(row=0, column=0, columnspan=4, sticky="w", pady=(0,10))

        self.symbol = tk.StringVar(value="ETHUSDT")
        self.interval = tk.StringVar(value="15m")
        self.side = tk.StringVar(value="LONG")
        self.s0 = tk.StringVar(value="0.01")
        self.r = tk.StringVar(value="1.5")
        self.max_steps = tk.StringVar(value="10")
        self.max_total = tk.StringVar(value="0.5")
        self.ldm = tk.StringVar(value="0")  # optional

        def row(i, label, var, widget="entry", values=None):
            ttk.Label(pad, text=label).grid(row=i, column=0, sticky="w")
            if widget == "combo":
                ttk.Combobox(pad, textvariable=var, values=values, width=12, state="readonly").grid(row=i, column=1, padx=(6,18))
            else:
                ttk.Entry(pad, textvariable=var, width=14).grid(row=i, column=1, padx=(6,18))

        row(1, "Coppia (Binance)", self.symbol)
        row(2, "Timeframe", self.interval, "combo", ["1m","3m","5m","15m","30m","1h","4h"])
        row(3, "Direzione", self.side, "combo", ["LONG","SHORT"])
        row(4, "Importo iniziale (ETH)", self.s0)
        row(5, "Fattore r", self.r)
        row(6, "Numero ricarichi", self.max_steps)
        row(7, "Max totale (ETH)", self.max_total)
        row(8, "LDM prezzo (opzionale)", self.ldm)

        ttk.Separator(pad).grid(row=9, column=0, columnspan=4, sticky="ew", pady=10)

        warn = ("LDM: se non lo sai, lascia 0.\n"
                "In quel caso EMDC usa un LDM 'di sicurezza' vicino alla BB.")
        ttk.Label(pad, text=warn, foreground="#444").grid(row=10, column=0, columnspan=4, sticky="w", pady=(0,10))

        btns = ttk.Frame(pad)
        btns.grid(row=11, column=0, columnspan=4, sticky="e")

        ttk.Button(btns, text="Annulla", command=self._cancel).pack(side="right")
        ttk.Button(btns, text="Calcola", command=self._ok).pack(side="right", padx=(0,8))

    def _cancel(self):
        self.result = None
        self.destroy()

    def _ok(self):
        try:
            self.result = {
                "symbol": self.symbol.get().strip().upper(),
                "interval": self.interval.get().strip(),
                "side": self.side.get().strip(),
                "s0": parse_float(self.s0.get(), "Importo iniziale"),
                "r": parse_float(self.r.get(), "r"),
                "max_steps": parse_int(self.max_steps.get(), "Numero ricarichi"),
                "max_total": parse_float(self.max_total.get(), "Max totale"),
                "ldm": parse_float(self.ldm.get(), "LDM"),
            }
            self.destroy()
        except Exception as e:
            messagebox.showerror("Errore", str(e))

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EMDC - Mamma Mode")
        self.geometry("1050x650")
        self.minsize(980, 620)

        if requests is None:
            messagebox.showerror("Dipendenza mancante", "Manca 'requests'. Installa: pip install requests")
            self.destroy()
            return

        self.plan: List[PlanRow] = []
        self.live: Optional[LiveSnapshot] = None
        self.cfg = None

        self._build_ui()
        self.after(100, self._run_wizard)

    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="EMDC - Versione semplice", font=("TkDefaultFont", 12, "bold")).pack(side="left")
        ttk.Button(top, text="Ricalcola (Wizard)", command=self._run_wizard).pack(side="right")

        mid = ttk.Frame(self, padding=(10,0,10,10))
        mid.pack(fill="both", expand=True)

        left = ttk.Frame(mid)
        left.pack(side="left", fill="both", expand=True)

        cols = ("step","size","total","price","note")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", height=18)
        self.tree.heading("step", text="Step")
        self.tree.heading("size", text="Size (ETH)")
        self.tree.heading("total", text="Totale (ETH)")
        self.tree.heading("price", text="Prezzo ricarico (USDT)")
        self.tree.heading("note", text="Note")
        self.tree.column("step", width=60, anchor="center")
        self.tree.column("size", width=120, anchor="e")
        self.tree.column("total", width=120, anchor="e")
        self.tree.column("price", width=160, anchor="e")
        self.tree.column("note", width=260, anchor="w")
        vsb = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        right = ttk.Frame(mid, padding=(12,0,0,0))
        right.pack(side="right", fill="y")

        ttk.Label(right, text="Riassunto", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        self.summary = ttk.Label(right, text="Nessun calcolo ancora.", wraplength=290, justify="left")
        self.summary.pack(anchor="w", pady=(6,12))

        ttk.Label(right, text="Grafico", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        ttk.Button(right, text="Apri grafico", command=self._plot).pack(anchor="w", pady=(6,0))

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=12)
        ttk.Label(right, text="Sicurezza", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        txt = ("• Non investire soldi che non puoi perdere.\n"
               "• Evita la leva alta.\n"
               "• Se rompe LDM: chiudi e stop.\n"
               "• Meglio iniziare in demo.")
        ttk.Label(right, text=txt, wraplength=290, justify="left").pack(anchor="w", pady=(6,0))

    def _run_wizard(self):
        wiz = Wizard(self)
        self.wait_window(wiz)
        if wiz.result is None:
            return
        self.cfg = wiz.result
        self._calculate_all()

    def _calculate_all(self):
        try:
            c = self.cfg
            live = get_live(c["symbol"], c["interval"])
            self.live = live

            # LDM: if not provided, set conservative near BB (slightly beyond)
            ldm = c["ldm"]
            if ldm == 0:
                if c["side"] == "LONG":
                    ldm = live.bb_lower * 0.995  # 0.5% below lower band
                else:
                    ldm = live.bb_upper * 1.005  # 0.5% above upper band

            plan = compute_plan(c["s0"], c["r"], c["max_steps"], c["max_total"], True, True)
            plan = auto_prices(plan, c["side"], live, ldm)
            self.plan = plan

            self._render_table()
            self._render_summary(ldm)
        except Exception as e:
            messagebox.showerror("Errore", str(e))

    def _render_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for r in self.plan:
            self.tree.insert("", "end", values=(
                r.step, f2(r.step_size), f2(r.total_size),
                "" if r.reload_price is None else f2(r.reload_price),
                r.note
            ))

    def _render_summary(self, ldm: float):
        c = self.cfg
        l = self.live
        total = max((r.total_size for r in self.plan), default=0.0)
        txt = (
            f"{c['symbol']} [{c['interval']}]\n"
            f"Direzione: {c['side']}\n"
            f"Prezzo: {f2(l.price)} USDT\n"
            f"BB: {l.bb_note}\n"
            f"RSI: {l.rsi14:.1f} | MACD: {l.macd_mode}\n"
            f"Trend: {l.adx_mode}\n"
            f"LDM: {f2(ldm)}\n\n"
            f"Importo iniziale: {f2(c['s0'])} ETH\n"
            f"r: {c['r']} | ricarichi: {c['max_steps']}\n"
            f"Max totale: {f2(c['max_total'])} ETH\n"
            f"Totale piano: {f2(total)} ETH"
        )
        self.summary.configure(text=txt)

    def _plot(self):
        if Figure is None or FigureCanvasTkAgg is None:
            messagebox.showerror("Grafico", "Manca matplotlib. Installa: pip install matplotlib")
            return
        if not self.plan:
            messagebox.showinfo("Grafico", "Prima calcola il piano.")
            return
        points = [(r.step, r.reload_price) for r in self.plan if r.step_size > 0 and r.reload_price is not None]
        if not points:
            messagebox.showinfo("Grafico", "Nessun prezzo da plottare.")
            return
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        live_price = self.live.price if self.live else None

        win = tk.Toplevel(self)
        win.title("Grafico - Prezzo vs Curva")
        win.geometry("900x520")

        fig = Figure(figsize=(8,4.5))
        ax = fig.add_subplot(111)
        ax.plot(x, y, marker="o")
        ax.set_xlabel("Step")
        ax.set_ylabel("Prezzo ricarico (USDT)")
        ax.grid(True, linestyle="--", linewidth=0.5)
        if live_price is not None:
            ax.axhline(live_price, linestyle="--")
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

def main():
    App().mainloop()

if __name__ == "__main__":
    main()
