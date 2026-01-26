
import time
import threading
from dataclasses import dataclass
from typing import Optional

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import ccxt

try:
    import winsound
except Exception:
    winsound = None


@dataclass
class Settings:
    exchange_id: str = "kucoin"
    symbol: str = "ETH/USDT"
    timeframe: str = "15m"
    limit: int = 800
    refresh_s: int = 15

    bb_len: int = 20
    bb_k: float = 2.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    comp_window: int = 300
    comp_thresh_pct: float = 10.0

    alert_breakout: bool = True
    alert_macd_hist_pos: bool = True
    alert_release: bool = True
    alert_containment: bool = False

    alert_cooldown_s: int = 120


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def compute_indicators(df: pd.DataFrame, stg: Settings) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]

    m = c.rolling(stg.bb_len).mean()
    sd = c.rolling(stg.bb_len).std(ddof=0)
    df["bb_mid"] = m
    df["bb_up"] = m + stg.bb_k * sd
    df["bb_dn"] = m - stg.bb_k * sd
    df["bb_width"] = (df["bb_up"] - df["bb_dn"]) / df["bb_mid"]

    fast = ema(c, stg.macd_fast)
    slow = ema(c, stg.macd_slow)
    df["macd"] = fast - slow
    df["macd_signal"] = ema(df["macd"], stg.macd_signal)
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    bw = df["bb_width"]

    def rolling_percentile(x):
        return float(x.rank(pct=True).iloc[-1] * 100.0)

    df["bb_width_pct"] = bw.rolling(stg.comp_window).apply(rolling_percentile, raw=False)

    N = 5
    df["macd_hist_slope"] = df["macd_hist"].diff().rolling(N).mean()
    df["near_mid"] = (df["close"] - df["bb_mid"]).abs() / df["bb_mid"]
    df["containment_flag"] = (
        (df["macd_hist_slope"] > 0) &
        (df["macd_hist"] < 0) &
        (df["near_mid"] < 0.002)
    )

    df["bb_width_rising"] = df["bb_width"].diff().rolling(3).mean() > 0
    df["release_flag"] = df["bb_width_rising"] & (df["close"] > df["bb_mid"])

    return df


def fetch_ohlcv(exchange_id: str, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    ex_class = getattr(ccxt, exchange_id, None)
    if ex_class is None:
        raise RuntimeError(f"Exchange non supportato: {exchange_id}")

    ex = ex_class({"enableRateLimit": True})
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    return df


class ScrollableFrame(ttk.Frame):
    """A vertically scrollable frame for dense control panels."""
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)

        self.inner.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        def _on_canvas_configure(event):
            # Make inner frame match canvas width
            self.canvas.itemconfigure(self.window_id, width=event.width)
        self.canvas.bind("<Configure>", _on_canvas_configure)

        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vsb.grid(row=0, column=1, sticky="ns")

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Mouse wheel support (Windows)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        # Windows: delta is 120 multiples
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ETH/USDT KuCoin — Compression & MACD Monitor")
        self.geometry("1250x760")
        self.minsize(980, 640)

        self.settings = Settings()
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._last_alert_ts = {}

        self._build_ui()

    def _build_ui(self):
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # LEFT (scrollable controls)
        left_container = ttk.Frame(self, padding=8)
        left_container.grid(row=0, column=0, sticky="ns")
        left_container.rowconfigure(0, weight=1)
        left_container.columnconfigure(0, weight=1)

        left = ScrollableFrame(left_container)
        left.grid(row=0, column=0, sticky="nsew")

        # RIGHT
        right = ttk.Frame(self, padding=10)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=2)
        right.rowconfigure(1, weight=1)
        right.rowconfigure(2, weight=1)
        right.columnconfigure(0, weight=1)

        # Populate LEFT controls inside left.inner
        panel = left.inner
        r = 0

        ttk.Label(panel, text="Controlli (scorri se non vedi tutto)", font=("Segoe UI", 12, "bold")).grid(row=r, column=0, sticky="w", pady=(0, 8)); r += 1

        self.exchange_var = tk.StringVar(value=self.settings.exchange_id)
        self.symbol_var = tk.StringVar(value=self.settings.symbol)
        self.tf_var = tk.StringVar(value=self.settings.timeframe)

        ttk.Label(panel, text="Exchange").grid(row=r, column=0, sticky="w"); r += 1
        self.exchange_cb = ttk.Combobox(panel, textvariable=self.exchange_var, values=["kucoin","binance","bybit","okx"], state="readonly")
        self.exchange_cb.grid(row=r, column=0, sticky="we", pady=(0, 8)); r += 1

        ttk.Label(panel, text="Symbol").grid(row=r, column=0, sticky="w"); r += 1
        ttk.Entry(panel, textvariable=self.symbol_var).grid(row=r, column=0, sticky="we", pady=(0, 8)); r += 1

        ttk.Label(panel, text="Timeframe").grid(row=r, column=0, sticky="w"); r += 1
        self.tf_cb = ttk.Combobox(panel, textvariable=self.tf_var, values=["1m","5m","15m","1h","4h","8h","1d"], state="readonly")
        self.tf_cb.grid(row=r, column=0, sticky="we", pady=(0, 8)); r += 1

        self.limit_var = tk.IntVar(value=self.settings.limit)
        self.refresh_var = tk.IntVar(value=self.settings.refresh_s)

        ttk.Label(panel, text="Candles (limit)").grid(row=r, column=0, sticky="w"); r += 1
        ttk.Spinbox(panel, from_=200, to=5000, increment=50, textvariable=self.limit_var).grid(row=r, column=0, sticky="we", pady=(0, 8)); r += 1

        ttk.Label(panel, text="Refresh (sec)").grid(row=r, column=0, sticky="w"); r += 1
        ttk.Spinbox(panel, from_=5, to=300, increment=5, textvariable=self.refresh_var).grid(row=r, column=0, sticky="we", pady=(0, 12)); r += 1

        ttk.Separator(panel).grid(row=r, column=0, sticky="we", pady=10); r += 1

        ttk.Label(panel, text="Indicatori", font=("Segoe UI", 12, "bold")).grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1

        self.bb_len_var = tk.IntVar(value=self.settings.bb_len)
        self.bb_k_var = tk.DoubleVar(value=self.settings.bb_k)
        self.mf_var = tk.IntVar(value=self.settings.macd_fast)
        self.ms_var = tk.IntVar(value=self.settings.macd_slow)
        self.msig_var = tk.IntVar(value=self.settings.macd_signal)

        ttk.Label(panel, text="BB length").grid(row=r, column=0, sticky="w"); r += 1
        ttk.Spinbox(panel, from_=10, to=200, increment=1, textvariable=self.bb_len_var).grid(row=r, column=0, sticky="we", pady=(0, 8)); r += 1

        ttk.Label(panel, text="BB stdev").grid(row=r, column=0, sticky="w"); r += 1
        ttk.Spinbox(panel, from_=1.0, to=4.0, increment=0.1, textvariable=self.bb_k_var).grid(row=r, column=0, sticky="we", pady=(0, 8)); r += 1

        ttk.Label(panel, text="MACD fast").grid(row=r, column=0, sticky="w"); r += 1
        ttk.Spinbox(panel, from_=5, to=30, increment=1, textvariable=self.mf_var).grid(row=r, column=0, sticky="we", pady=(0, 8)); r += 1

        ttk.Label(panel, text="MACD slow").grid(row=r, column=0, sticky="w"); r += 1
        ttk.Spinbox(panel, from_=10, to=60, increment=1, textvariable=self.ms_var).grid(row=r, column=0, sticky="we", pady=(0, 8)); r += 1

        ttk.Label(panel, text="MACD signal").grid(row=r, column=0, sticky="w"); r += 1
        ttk.Spinbox(panel, from_=3, to=30, increment=1, textvariable=self.msig_var).grid(row=r, column=0, sticky="we", pady=(0, 12)); r += 1

        ttk.Separator(panel).grid(row=r, column=0, sticky="we", pady=10); r += 1

        ttk.Label(panel, text="Compressione", font=("Segoe UI", 12, "bold")).grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1

        self.comp_window_var = tk.IntVar(value=self.settings.comp_window)
        self.comp_thresh_var = tk.DoubleVar(value=self.settings.comp_thresh_pct)

        ttk.Label(panel, text="Window (pct)").grid(row=r, column=0, sticky="w"); r += 1
        ttk.Spinbox(panel, from_=50, to=2000, increment=10, textvariable=self.comp_window_var).grid(row=r, column=0, sticky="we", pady=(0, 8)); r += 1

        ttk.Label(panel, text="Threshold pct").grid(row=r, column=0, sticky="w"); r += 1
        ttk.Spinbox(panel, from_=1, to=30, increment=1, textvariable=self.comp_thresh_var).grid(row=r, column=0, sticky="we", pady=(0, 12)); r += 1

        ttk.Separator(panel).grid(row=r, column=0, sticky="we", pady=10); r += 1

        ttk.Label(panel, text="Alert", font=("Segoe UI", 12, "bold")).grid(row=r, column=0, sticky="w", pady=(0, 6)); r += 1

        self.alert_breakout_var = tk.BooleanVar(value=self.settings.alert_breakout)
        self.alert_macd0_var = tk.BooleanVar(value=self.settings.alert_macd_hist_pos)
        self.alert_release_var = tk.BooleanVar(value=self.settings.alert_release)
        self.alert_containment_var = tk.BooleanVar(value=self.settings.alert_containment)

        ttk.Checkbutton(panel, text="Breakout > Upper BB", variable=self.alert_breakout_var).grid(row=r, column=0, sticky="w"); r += 1
        ttk.Checkbutton(panel, text="MACD hist > 0", variable=self.alert_macd0_var).grid(row=r, column=0, sticky="w"); r += 1
        ttk.Checkbutton(panel, text="Release flag", variable=self.alert_release_var).grid(row=r, column=0, sticky="w"); r += 1
        ttk.Checkbutton(panel, text="Containment flag", variable=self.alert_containment_var).grid(row=r, column=0, sticky="w"); r += 1

        self.start_btn = ttk.Button(panel, text="Start", command=self.start)
        self.stop_btn = ttk.Button(panel, text="Stop", command=self.stop, state="disabled")
        self.start_btn.grid(row=r, column=0, sticky="we", pady=(14, 6)); r += 1
        self.stop_btn.grid(row=r, column=0, sticky="we"); r += 1

        self.status_var = tk.StringVar(value="Pronto.")
        ttk.Label(panel, textvariable=self.status_var, wraplength=260).grid(row=r, column=0, sticky="we", pady=(10, 0)); r += 1

        panel.columnconfigure(0, weight=1)

        # RIGHT: charts
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax_price = self.fig.add_subplot(211)
        self.ax_macd = self.fig.add_subplot(212)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Table
        self.table = ttk.Treeview(right, columns=("ts","close","bb_pct","bb_w","macd_h","flags"), show="headings", height=8)
        for c, w in [("ts",170),("close",90),("bb_pct",80),("bb_w",80),("macd_h",90),("flags",240)]:
            self.table.heading(c, text=c)
            self.table.column(c, width=w, anchor="w")
        self.table.grid(row=1, column=0, sticky="nsew", pady=(10, 6))

        # Log
        self.log = tk.Text(right, height=8, wrap="word")
        self.log.grid(row=2, column=0, sticky="nsew")
        self._log("⚙️ Default: KuCoin ETH/USDT 15m. Premi Start. (Pannello sinistro scrollabile)\n")

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.log.insert("end", f"[{ts}] {msg}\n")
        self.log.see("end")

    def _read_settings_from_ui(self) -> Settings:
        s = Settings(
            exchange_id=self.exchange_var.get().strip(),
            symbol=self.symbol_var.get().strip(),
            timeframe=self.tf_var.get().strip(),
            limit=int(self.limit_var.get()),
            refresh_s=int(self.refresh_var.get()),
            bb_len=int(self.bb_len_var.get()),
            bb_k=float(self.bb_k_var.get()),
            macd_fast=int(self.mf_var.get()),
            macd_slow=int(self.ms_var.get()),
            macd_signal=int(self.msig_var.get()),
            comp_window=int(self.comp_window_var.get()),
            comp_thresh_pct=float(self.comp_thresh_var.get()),
            alert_breakout=bool(self.alert_breakout_var.get()),
            alert_macd_hist_pos=bool(self.alert_macd0_var.get()),
            alert_release=bool(self.alert_release_var.get()),
            alert_containment=bool(self.alert_containment_var.get()),
        )
        if s.macd_fast >= s.macd_slow:
            raise ValueError("MACD fast deve essere < MACD slow")
        return s

    def start(self):
        try:
            self.settings = self._read_settings_from_ui()
        except Exception as e:
            messagebox.showerror("Impostazioni non valide", str(e))
            return

        if self._worker_thread and self._worker_thread.is_alive():
            return

        self._stop_event.clear()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_var.set("Running…")
        self._log(f"▶ Start: {self.settings.exchange_id} {self.settings.symbol} {self.settings.timeframe} refresh={self.settings.refresh_s}s")

        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def stop(self):
        self._stop_event.set()
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_var.set("Stopped.")
        self._log("⏹ Stop")

    def _on_close(self):
        self._stop_event.set()
        self.destroy()

    def _beep(self):
        if winsound is not None:
            try:
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                return
            except Exception:
                pass
            try:
                winsound.Beep(880, 200)
            except Exception:
                pass

    def _cooldown_ok(self, key: str) -> bool:
        now = time.time()
        last = self._last_alert_ts.get(key, 0)
        if now - last >= self.settings.alert_cooldown_s:
            self._last_alert_ts[key] = now
            return True
        return False

    def _alert(self, title: str, body: str, key: str):
        if not self._cooldown_ok(key):
            return
        self._beep()
        self._log(f"🔔 {title} — {body}")
        self.after(0, lambda: messagebox.showinfo(title, body))

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                df = fetch_ohlcv(self.settings.exchange_id, self.settings.symbol, self.settings.timeframe, self.settings.limit)
                df = compute_indicators(df, self.settings)
                self.after(0, lambda d=df: self._render(d))
            except Exception as e:
                self.after(0, lambda: self._log(f"❌ Errore: {e}"))
                self.after(0, lambda: self.status_var.set(f"Errore: {e}"))

            for _ in range(max(1, self.settings.refresh_s)):
                if self._stop_event.is_set():
                    break
                time.sleep(1)

    def _render(self, df: pd.DataFrame):
        df2 = df.dropna().copy()
        if df2.empty:
            self.status_var.set("Dati insufficienti (dropna).")
            return

        last = df2.iloc[-1]
        price = float(last["close"])
        bb_pct = float(last["bb_width_pct"]) if not np.isnan(last["bb_width_pct"]) else np.nan
        state = "COMPRESSIONE" if (not np.isnan(bb_pct) and bb_pct <= self.settings.comp_thresh_pct) else "NORMALE"
        self.status_var.set(f"Prezzo {price:.2f} | BB width pct {bb_pct:.1f} | {state}")

        t = df2["timestamp"]

        self.ax_price.clear()
        self.ax_macd.clear()

        self.ax_price.plot(t, df2["close"], label="close")
        self.ax_price.plot(t, df2["bb_up"], label="bb_up")
        self.ax_price.plot(t, df2["bb_mid"], label="bb_mid")
        self.ax_price.plot(t, df2["bb_dn"], label="bb_dn")
        self.ax_price.set_title("Prezzo + Bollinger")
        self.ax_price.legend(loc="upper left", fontsize=8)

        self.ax_macd.plot(t, df2["macd"], label="macd")
        self.ax_macd.plot(t, df2["macd_signal"], label="signal")
        self.ax_macd.bar(t, df2["macd_hist"], width=0.01, label="hist")
        self.ax_macd.set_title("MACD")
        self.ax_macd.legend(loc="upper left", fontsize=8)

        self.fig.tight_layout()
        self.canvas.draw()

        for r in self.table.get_children():
            self.table.delete(r)
        tail = df2.tail(20)
        for _, row in tail.iterrows():
            flags = []
            if bool(row.get("containment_flag", False)):
                flags.append("containment")
            if bool(row.get("release_flag", False)):
                flags.append("release")
            self.table.insert("", "end", values=(
                row["timestamp"].strftime("%Y-%m-%d %H:%M"),
                f"{row['close']:.2f}",
                f"{row['bb_width_pct']:.1f}" if not np.isnan(row["bb_width_pct"]) else "",
                f"{row['bb_width']*100:.3f}%",
                f"{row['macd_hist']:.4f}",
                ",".join(flags)
            ))

        if self.settings.alert_breakout and float(last["close"]) > float(last["bb_up"]):
            self._alert("Breakout sopra Upper BB",
                        f"Close {last['close']:.2f} > UpperBB {last['bb_up']:.2f}",
                        key="breakout_up")

        if self.settings.alert_macd_hist_pos and float(last["macd_hist"]) > 0:
            self._alert("MACD histogram > 0",
                        f"MACD_hist {last['macd_hist']:.4f} (possibile cambio momentum)",
                        key="macd_hist_pos")

        if self.settings.alert_release and bool(last.get("release_flag", False)):
            self._alert("Release flag",
                        "BB width in risalita + close sopra BB mid (possibile rilascio compressione)",
                        key="release")

        if self.settings.alert_containment and bool(last.get("containment_flag", False)):
            self._alert("Containment flag",
                        "MACD hist in salita ma ancora < 0 + prezzo vicino BB mid (possibile 'tenuta' negativa)",
                        key="containment")


if __name__ == "__main__":
    App().mainloop()
