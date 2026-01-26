#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETH Liquidation Triple Dashboard (Binance + Bybit + KuCoin)
Single-file Windows-friendly GUI (Tkinter) with:
- Real-time liquidation feed (Binance Futures, Bybit Linear) when available
- KuCoin "proxy flush" detector (large trade sweeps + volatility bursts) because KuCoin does not expose a
  widely documented public liquidation tape in the same way (this tab is clearly labeled as a proxy).
- Order book imbalance + "book is full" style liquidity meter (top-N depth)
- Open Interest + Funding snapshots (REST polling)
- Sound + popup alerts on thresholds

DISCLAIMER
---------
This tool is for monitoring/education only. It is NOT financial advice and cannot guarantee correctness.
Exchange APIs can change; if a stream breaks, check the "Status" panel for errors and update endpoints.

Tested conceptually on Python 3.10+.
Dependencies (pip):
    pip install requests websocket-client numpy matplotlib

Notes:
- Uses background threads for websockets + REST poller.
- Uses Tkinter + matplotlib FigureCanvasTkAgg.
"""
from __future__ import annotations

import json
import math
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests
import numpy as np

# websocket-client (not "websockets" asyncio) keeps Tkinter integration simpler
from websocket import WebSocketApp

import tkinter as tk
from tkinter import ttk, messagebox

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ---------------------------
# Utilities
# ---------------------------

def now_ts() -> float:
    return time.time()

def fmt_ts(ts: float) -> str:
    # local time
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))

def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def beep():
    # Windows: winsound; others: Tk bell
    try:
        import winsound
        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
    except Exception:
        try:
            tk._default_root.bell()
        except Exception:
            pass

# ---------------------------
# Data models
# ---------------------------

@dataclass
class LiqEvent:
    ts: float
    exchange: str
    symbol: str
    side: str               # "BUY"/"SELL" or "LongLiq"/"ShortLiq"
    price: float
    qty: float
    notional: float
    raw: dict = field(default_factory=dict)

@dataclass
class OIEntry:
    ts: float
    exchange: str
    symbol: str
    open_interest: float
    funding_rate: Optional[float] = None

@dataclass
class BookSnapshot:
    ts: float
    exchange: str
    symbol: str
    bid: float
    ask: float
    bid_depth: float
    ask_depth: float
    imbalance: float        # (bid-ask)/(bid+ask)
    spread_bps: float

# ---------------------------
# Exchange connectors
# ---------------------------

class BaseConnector:
    def __init__(self, symbol: str, out_q: queue.Queue, status_cb):
        self.symbol = symbol
        self.out_q = out_q
        self.status_cb = status_cb
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def stopped(self) -> bool:
        return self._stop.is_set()

# ---- Binance Futures (USDT-M) ----

class BinanceConnector(BaseConnector):
    REST = "https://fapi.binance.com"
    WS   = "wss://fstream.binance.com/ws"

    def __init__(self, symbol: str, out_q: queue.Queue, status_cb, book_depth: int = 50):
        super().__init__(symbol, out_q, status_cb)
        self.book_depth = int(book_depth)
        self.ws_thread = None
        self.ws = None

    def _ws_url(self) -> str:
        # Liquidations tape: forceOrder
        s = self.symbol.lower()
        return f"{self.WS}/{s}@forceOrder"

    def start_ws(self):
        def on_message(ws, msg):
            try:
                data = json.loads(msg)
                if "o" not in data:
                    return
                o = data["o"]
                # Example fields: s, S, o, f, q, p, ap, X, l, z, T
                symbol = o.get("s", self.symbol)
                side = o.get("S", "")
                # Binance uses S=BUY/SELL (liquidation direction of order placed to close)
                price = safe_float(o.get("ap") or o.get("p") or 0.0, 0.0)
                qty = safe_float(o.get("q") or 0.0, 0.0)
                notional = price * qty
                ev = LiqEvent(ts=now_ts(), exchange="Binance", symbol=symbol, side=side,
                              price=price, qty=qty, notional=notional, raw=data)
                self.out_q.put(("liq", ev))
            except Exception:
                self.status_cb("Binance WS parse error:\n" + traceback.format_exc())
        def on_error(ws, err):
            self.status_cb(f"Binance WS error: {err}")
        def on_close(ws, code, reason):
            self.status_cb(f"Binance WS closed: {code} {reason}")
        def on_open(ws):
            self.status_cb("Binance WS connected.")
        def run():
            while not self.stopped():
                try:
                    self.ws = WebSocketApp(self._ws_url(),
                                           on_open=on_open, on_message=on_message,
                                           on_error=on_error, on_close=on_close)
                    self.ws.run_forever(ping_interval=20, ping_timeout=10)
                except Exception as e:
                    self.status_cb(f"Binance WS exception: {e}")
                if self.stopped():
                    break
                time.sleep(3)
        self.ws_thread = threading.Thread(target=run, daemon=True)
        self.ws_thread.start()

    def fetch_book(self) -> Optional[BookSnapshot]:
        try:
            r = requests.get(f"{self.REST}/fapi/v1/depth",
                             params={"symbol": self.symbol, "limit": min(max(self.book_depth, 5), 1000)},
                             timeout=10)
            r.raise_for_status()
            j = r.json()
            bids = j.get("bids", [])
            asks = j.get("asks", [])
            if not bids or not asks:
                return None
            best_bid = float(bids[0][0]); best_ask = float(asks[0][0])
            bid_depth = sum(float(px)*float(q) for px, q in bids[:self.book_depth])
            ask_depth = sum(float(px)*float(q) for px, q in asks[:self.book_depth])
            imb = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-12)
            spread_bps = (best_ask - best_bid) / ((best_ask + best_bid)/2) * 1e4
            return BookSnapshot(ts=now_ts(), exchange="Binance", symbol=self.symbol,
                                bid=best_bid, ask=best_ask, bid_depth=bid_depth, ask_depth=ask_depth,
                                imbalance=imb, spread_bps=spread_bps)
        except Exception as e:
            self.status_cb(f"Binance book error: {e}")
            return None

    def fetch_oi_funding(self) -> Optional[OIEntry]:
        try:
            oi_r = requests.get(f"{self.REST}/fapi/v1/openInterest",
                                params={"symbol": self.symbol}, timeout=10)
            oi_r.raise_for_status()
            oi = safe_float(oi_r.json().get("openInterest"), None)
            fr_r = requests.get(f"{self.REST}/fapi/v1/fundingRate",
                                params={"symbol": self.symbol, "limit": 1}, timeout=10)
            fr = None
            if fr_r.ok:
                arr = fr_r.json()
                if isinstance(arr, list) and arr:
                    fr = safe_float(arr[-1].get("fundingRate"), None)
            if oi is None:
                return None
            return OIEntry(ts=now_ts(), exchange="Binance", symbol=self.symbol,
                           open_interest=oi, funding_rate=fr)
        except Exception as e:
            self.status_cb(f"Binance OI/funding error: {e}")
            return None

# ---- Bybit (v5 public linear) ----

class BybitConnector(BaseConnector):
    REST = "https://api.bybit.com"
    WS   = "wss://stream.bybit.com/v5/public/linear"

    def __init__(self, symbol: str, out_q: queue.Queue, status_cb, book_depth: int = 50):
        super().__init__(symbol, out_q, status_cb)
        self.book_depth = int(book_depth)
        self.ws_thread = None
        self.ws = None
        self._subbed = False

    def start_ws(self):
        # Bybit v5 websocket requires subscription message after open
        def on_open(ws):
            try:
                sub = {"op": "subscribe", "args": [f"liquidation.{self.symbol}", f"orderbook.50.{self.symbol}", f"tickers.{self.symbol}"]}
                ws.send(json.dumps(sub))
                self.status_cb("Bybit WS connected + subscribed.")
                self._subbed = True
            except Exception as e:
                self.status_cb(f"Bybit WS subscribe error: {e}")

        def on_message(ws, msg):
            try:
                data = json.loads(msg)
                topic = data.get("topic", "")
                if topic.startswith("liquidation."):
                    # data["data"] can be list
                    items = data.get("data", [])
                    if isinstance(items, dict):
                        items = [items]
                    for it in items:
                        side = it.get("side", "")
                        price = safe_float(it.get("price"), 0.0)
                        qty = safe_float(it.get("size") or it.get("qty"), 0.0)
                        notional = price * qty
                        ev = LiqEvent(ts=now_ts(), exchange="Bybit", symbol=self.symbol,
                                      side=side, price=price, qty=qty, notional=notional, raw=it)
                        self.out_q.put(("liq", ev))
                elif topic.startswith("orderbook."):
                    # We'll process orderbook for depth/imbalance
                    d = data.get("data", {})
                    bids = d.get("b", [])  # [price, size]
                    asks = d.get("a", [])
                    if bids and asks:
                        best_bid = float(bids[0][0]); best_ask = float(asks[0][0])
                        bid_depth = sum(float(px)*float(q) for px, q in bids[:self.book_depth])
                        ask_depth = sum(float(px)*float(q) for px, q in asks[:self.book_depth])
                        imb = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-12)
                        spread_bps = (best_ask - best_bid) / ((best_ask + best_bid)/2) * 1e4
                        snap = BookSnapshot(ts=now_ts(), exchange="Bybit", symbol=self.symbol,
                                            bid=best_bid, ask=best_ask, bid_depth=bid_depth, ask_depth=ask_depth,
                                            imbalance=imb, spread_bps=spread_bps)
                        self.out_q.put(("book", snap))
                elif topic.startswith("tickers."):
                    # ignore; we use it as keepalive/price display if needed
                    pass
            except Exception:
                self.status_cb("Bybit WS parse error:\n" + traceback.format_exc())

        def on_error(ws, err):
            self.status_cb(f"Bybit WS error: {err}")

        def on_close(ws, code, reason):
            self.status_cb(f"Bybit WS closed: {code} {reason}")
            self._subbed = False

        def run():
            while not self.stopped():
                try:
                    self.ws = WebSocketApp(self.WS, on_open=on_open, on_message=on_message,
                                           on_error=on_error, on_close=on_close)
                    self.ws.run_forever(ping_interval=20, ping_timeout=10)
                except Exception as e:
                    self.status_cb(f"Bybit WS exception: {e}")
                if self.stopped():
                    break
                time.sleep(3)

        self.ws_thread = threading.Thread(target=run, daemon=True)
        self.ws_thread.start()

    def fetch_book_rest(self) -> Optional[BookSnapshot]:
        # fallback if ws book fails
        try:
            r = requests.get(f"{self.REST}/v5/market/orderbook",
                             params={"category": "linear", "symbol": self.symbol, "limit": min(max(self.book_depth, 5), 50)},
                             timeout=10)
            r.raise_for_status()
            j = r.json()
            result = j.get("result", {})
            bids = result.get("b", [])
            asks = result.get("a", [])
            if not bids or not asks:
                return None
            best_bid = float(bids[0][0]); best_ask = float(asks[0][0])
            bid_depth = sum(float(px)*float(q) for px, q in bids[:self.book_depth])
            ask_depth = sum(float(px)*float(q) for px, q in asks[:self.book_depth])
            imb = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-12)
            spread_bps = (best_ask - best_bid) / ((best_ask + best_bid)/2) * 1e4
            return BookSnapshot(ts=now_ts(), exchange="Bybit", symbol=self.symbol,
                                bid=best_bid, ask=best_ask, bid_depth=bid_depth, ask_depth=ask_depth,
                                imbalance=imb, spread_bps=spread_bps)
        except Exception as e:
            self.status_cb(f"Bybit book error: {e}")
            return None

    def fetch_oi_funding(self) -> Optional[OIEntry]:
        try:
            # Open interest
            oi_r = requests.get(f"{self.REST}/v5/market/open-interest",
                                params={"category": "linear", "symbol": self.symbol, "intervalTime": "5min"},
                                timeout=10)
            oi_r.raise_for_status()
            j = oi_r.json()
            result = j.get("result", {})
            lst = result.get("list", [])
            oi = None
            if lst:
                oi = safe_float(lst[0].get("openInterest"), None)
            # Funding rate
            fr_r = requests.get(f"{self.REST}/v5/market/funding/history",
                                params={"category":"linear", "symbol": self.symbol, "limit": 1},
                                timeout=10)
            fr = None
            if fr_r.ok:
                jj = fr_r.json()
                rr = jj.get("result", {})
                ll = rr.get("list", [])
                if ll:
                    fr = safe_float(ll[0].get("fundingRate"), None)
            if oi is None:
                return None
            return OIEntry(ts=now_ts(), exchange="Bybit", symbol=self.symbol,
                           open_interest=oi, funding_rate=fr)
        except Exception as e:
            self.status_cb(f"Bybit OI/funding error: {e}")
            return None

# ---- KuCoin Futures (public REST + WS for trades) ----

class KucoinConnector(BaseConnector):
    REST = "https://api-futures.kucoin.com"

    def __init__(self, symbol: str, out_q: queue.Queue, status_cb, book_depth: int = 50):
        super().__init__(symbol, out_q, status_cb)
        self.book_depth = int(book_depth)
        self.ws_thread = None
        self.ws = None

        # For "proxy flush" detection
        self._last_price = None
        self._recent_trades: List[Tuple[float, float, float]] = []  # (ts, price, size)
        self._lock = threading.Lock()

    def _get_ws_token(self) -> Optional[dict]:
        try:
            r = requests.post(f"{self.REST}/api/v1/bullet-public", timeout=10)
            r.raise_for_status()
            j = r.json()
            if j.get("code") != "200000":
                return None
            return j.get("data")
        except Exception as e:
            self.status_cb(f"KuCoin token error: {e}")
            return None

    def start_ws(self):
        # KuCoin requires token to get endpoint + ping interval.
        def run():
            while not self.stopped():
                data = self._get_ws_token()
                if not data:
                    time.sleep(5)
                    continue
                endpoint = data.get("instanceServers", [{}])[0].get("endpoint")
                token = data.get("token")
                ping_interval = data.get("instanceServers", [{}])[0].get("pingInterval", 20000) / 1000.0
                if not endpoint or not token:
                    time.sleep(5)
                    continue
                url = f"{endpoint}?token={token}"
                topic = f"/contractMarket/execution:{self.symbol}"

                def on_open(ws):
                    try:
                        sub = {"id": str(int(time.time()*1000)), "type":"subscribe", "topic": topic, "privateChannel": False, "response": True}
                        ws.send(json.dumps(sub))
                        self.status_cb("KuCoin WS connected + subscribed (trades).")
                    except Exception as e:
                        self.status_cb(f"KuCoin WS subscribe error: {e}")

                def on_message(ws, msg):
                    try:
                        j = json.loads(msg)
                        if j.get("type") == "message" and j.get("topic","").startswith("/contractMarket/execution:"):
                            d = j.get("data", {})
                            price = safe_float(d.get("price"), None)
                            size = safe_float(d.get("size"), None)
                            side = d.get("side", "")
                            if price is None or size is None:
                                return
                            ts = now_ts()
                            # Maintain recent trade list for proxy detection
                            with self._lock:
                                self._recent_trades.append((ts, price, size if side.lower()=="buy" else -size))
                                # keep 5 minutes
                                cutoff = ts - 300
                                self._recent_trades = [t for t in self._recent_trades if t[0] >= cutoff]
                                self._last_price = price
                            # Proxy: large sweep detection
                            abs_notional = abs(price*size)
                            self.out_q.put(("ku_trade", {"ts":ts, "price":price, "size":size, "side":side, "notional":abs_notional}))
                    except Exception:
                        self.status_cb("KuCoin WS parse error:\n" + traceback.format_exc())

                def on_error(ws, err):
                    self.status_cb(f"KuCoin WS error: {err}")

                def on_close(ws, code, reason):
                    self.status_cb(f"KuCoin WS closed: {code} {reason}")

                def ping_loop(ws):
                    while not self.stopped():
                        try:
                            ws.send(json.dumps({"id": str(int(time.time()*1000)), "type":"ping"}))
                        except Exception:
                            break
                        time.sleep(max(5, ping_interval*0.9))

                try:
                    self.ws = WebSocketApp(url, on_open=on_open, on_message=on_message,
                                           on_error=on_error, on_close=on_close)
                    ping_thread = threading.Thread(target=ping_loop, args=(self.ws,), daemon=True)
                    ping_thread.start()
                    self.ws.run_forever(ping_interval=0)  # we handle ping
                except Exception as e:
                    self.status_cb(f"KuCoin WS exception: {e}")
                if self.stopped():
                    break
                time.sleep(3)

        self.ws_thread = threading.Thread(target=run, daemon=True)
        self.ws_thread.start()

    def fetch_book(self) -> Optional[BookSnapshot]:
        try:
            # KuCoin Futures Level2 depth
            r = requests.get(f"{self.REST}/api/v1/level2/depth{''}",
                             params={"symbol": self.symbol}, timeout=10)
            # Note: Endpoint can vary; if this fails, fall back to /api/v1/level2/snapshot
            if r.status_code != 200:
                r = requests.get(f"{self.REST}/api/v1/level2/snapshot", params={"symbol": self.symbol}, timeout=10)
            r.raise_for_status()
            j = r.json()
            if j.get("code") != "200000":
                return None
            data = j.get("data", {})
            bids = data.get("bids", [])
            asks = data.get("asks", [])
            if not bids or not asks:
                return None
            # bids/asks: [[price, size], ...] or [{"price":"", "size":""}, ...]
            def norm_side(arr):
                out = []
                for it in arr:
                    if isinstance(it, (list, tuple)) and len(it) >= 2:
                        out.append((float(it[0]), float(it[1])))
                    elif isinstance(it, dict):
                        out.append((float(it.get("price")), float(it.get("size"))))
                return out
            bids = norm_side(bids); asks = norm_side(asks)
            bids.sort(key=lambda x: -x[0]); asks.sort(key=lambda x: x[0])
            best_bid = bids[0][0]; best_ask = asks[0][0]
            bid_depth = sum(px*q for px,q in bids[:self.book_depth])
            ask_depth = sum(px*q for px,q in asks[:self.book_depth])
            imb = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-12)
            spread_bps = (best_ask - best_bid) / ((best_ask + best_bid)/2) * 1e4
            return BookSnapshot(ts=now_ts(), exchange="KuCoin", symbol=self.symbol,
                                bid=best_bid, ask=best_ask, bid_depth=bid_depth, ask_depth=ask_depth,
                                imbalance=imb, spread_bps=spread_bps)
        except Exception as e:
            self.status_cb(f"KuCoin book error: {e}")
            return None

    def fetch_oi_funding(self) -> Optional[OIEntry]:
        try:
            # Open interest + funding (mark) endpoints
            oi_r = requests.get(f"{self.REST}/api/v1/openInterest",
                                params={"symbol": self.symbol}, timeout=10)
            oi = None
            if oi_r.ok:
                j = oi_r.json()
                if j.get("code") == "200000":
                    oi = safe_float(j.get("data", {}).get("value"), None)
            fr = None
            fr_r = requests.get(f"{self.REST}/api/v1/funding-rate/{self.symbol}", timeout=10)
            if fr_r.ok:
                jj = fr_r.json()
                if jj.get("code") == "200000":
                    fr = safe_float(jj.get("data", {}).get("value"), None)
            if oi is None:
                return None
            return OIEntry(ts=now_ts(), exchange="KuCoin", symbol=self.symbol,
                           open_interest=oi, funding_rate=fr)
        except Exception as e:
            self.status_cb(f"KuCoin OI/funding error: {e}")
            return None

    def compute_proxy_flush(self, window_s: int = 120, notional_thresh: float = 1_000_000.0,
                            vol_z: float = 3.0) -> Optional[LiqEvent]:
        """
        Proxy liquidation/flush detector using large signed notional & volatility burst.
        Not a real liquidation tape; used only as 'stress / sweep' signal.
        """
        with self._lock:
            if not self._recent_trades:
                return None
            ts = now_ts()
            trades = [t for t in self._recent_trades if t[0] >= ts - window_s]
        if len(trades) < 20:
            return None

        prices = np.array([t[1] for t in trades], dtype=float)
        signed_sizes = np.array([t[2] for t in trades], dtype=float)
        # signed notional (approx)
        signed_notional = float(np.sum(prices * signed_sizes))
        abs_notional = float(np.sum(np.abs(prices * signed_sizes)))

        # volatility burst proxy
        rets = np.diff(prices) / (prices[:-1] + 1e-12)
        vol = float(np.std(rets))
        # compare with longer baseline
        with self._lock:
            long_trades = list(self._recent_trades)
        long_prices = np.array([t[1] for t in long_trades], dtype=float)
        if len(long_prices) > 50:
            long_rets = np.diff(long_prices[-300:]) / (long_prices[-301:-1] + 1e-12) if len(long_prices) > 301 else np.diff(long_prices) / (long_prices[:-1] + 1e-12)
            base_vol = float(np.std(long_rets)) + 1e-12
        else:
            base_vol = vol + 1e-12
        z = vol / base_vol

        if abs_notional >= notional_thresh and z >= vol_z:
            side = "SELL" if signed_notional < 0 else "BUY"
            price = float(prices[-1])
            qty = float(abs_notional / max(price, 1e-9))
            ev = LiqEvent(ts=ts, exchange="KuCoin (proxy)", symbol=self.symbol,
                          side=side, price=price, qty=qty, notional=abs_notional,
                          raw={"window_s": window_s, "abs_notional": abs_notional, "vol_z": z})
            return ev
        return None

# ---------------------------
# Dashboard / App
# ---------------------------

class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ETH Liquidation Triple Dashboard (Binance + Bybit + KuCoin)")
        self.root.geometry("1220x760")

        self.q = queue.Queue()

        self.status_lines: List[str] = []
        self.max_status = 200

        # Default symbols:
        # Binance/Bybit use ETHUSDT. KuCoin futures commonly uses ETHUSDTM (perp).
        self.sym_binance = tk.StringVar(value="ETHUSDT")
        self.sym_bybit   = tk.StringVar(value="ETHUSDT")
        self.sym_kucoin  = tk.StringVar(value="ETHUSDTM")

        self.depth_n = tk.IntVar(value=50)

        # Alerts
        self.alert_notional = tk.DoubleVar(value=500_000.0)
        self.alert_imbalance = tk.DoubleVar(value=0.30)
        self.alert_spread_bps = tk.DoubleVar(value=8.0)
        self.proxy_notional = tk.DoubleVar(value=1_000_000.0)
        self.proxy_volz = tk.DoubleVar(value=3.0)
        self.alert_enabled = tk.BooleanVar(value=True)

        # Data stores
        self.liq_events: List[LiqEvent] = []
        self.max_liq = 600

        self.book: Dict[str, BookSnapshot] = {}
        self.oi: Dict[str, OIEntry] = {}

        self.ku_proxy_events: List[LiqEvent] = []
        self.max_proxy = 300

        # connectors
        self.bin = None
        self.byb = None
        self.kuc = None

        self._threads: List[threading.Thread] = []
        self._running = False

        self._build_ui()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # UI update loop
        self.root.after(250, self._drain_queue)
        self.root.after(1000, self._refresh_charts)

    def log_status(self, line: str):
        line = f"[{fmt_ts(now_ts())}] {line}"
        self.status_lines.append(line)
        if len(self.status_lines) > self.max_status:
            self.status_lines = self.status_lines[-self.max_status:]
        if hasattr(self, "txt_status"):
            self.txt_status.configure(state="normal")
            self.txt_status.delete("1.0", "end")
            self.txt_status.insert("end", "\n".join(self.status_lines[-60:]))
            self.txt_status.configure(state="disabled")

    def _build_ui(self):
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=10, pady=8)

        # Controls
        def lbl(parent, text):
            w = ttk.Label(parent, text=text)
            w.pack(side="left", padx=(0,6))
            return w

        lbl(top, "Binance:")
        ttk.Entry(top, textvariable=self.sym_binance, width=10).pack(side="left", padx=(0,10))
        lbl(top, "Bybit:")
        ttk.Entry(top, textvariable=self.sym_bybit, width=10).pack(side="left", padx=(0,10))
        lbl(top, "KuCoin:")
        ttk.Entry(top, textvariable=self.sym_kucoin, width=10).pack(side="left", padx=(0,10))

        lbl(top, "Depth N:")
        ttk.Spinbox(top, from_=5, to=100, textvariable=self.depth_n, width=6).pack(side="left", padx=(0,10))

        ttk.Checkbutton(top, text="Alerts", variable=self.alert_enabled).pack(side="left", padx=(0,10))

        ttk.Button(top, text="Start", command=self.start).pack(side="left", padx=(0,8))
        ttk.Button(top, text="Stop", command=self.stop).pack(side="left", padx=(0,8))
        ttk.Button(top, text="Test Alert", command=self.test_alert).pack(side="left", padx=(0,8))

        # Notebook tabs
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=10, pady=8)
        self.nb = nb

        self.tab_overview = ttk.Frame(nb)
        self.tab_liq = ttk.Frame(nb)
        self.tab_book = ttk.Frame(nb)
        self.tab_macro = ttk.Frame(nb)
        self.tab_status = ttk.Frame(nb)

        nb.add(self.tab_overview, text="Overview")
        nb.add(self.tab_liq, text="Liquidations")
        nb.add(self.tab_book, text="Order Book / Liquidity")
        nb.add(self.tab_macro, text="OI + Funding")
        nb.add(self.tab_status, text="Status / Settings")

        self._build_overview()
        self._build_liq()
        self._build_book()
        self._build_macro()
        self._build_status()

    def _build_overview(self):
        f = self.tab_overview
        top = ttk.Frame(f)
        top.pack(fill="x", padx=10, pady=8)

        self.lbl_price = ttk.Label(top, text="—", font=("Segoe UI", 14, "bold"))
        self.lbl_price.pack(side="left")

        self.lbl_book = ttk.Label(top, text=" ", font=("Segoe UI", 11))
        self.lbl_book.pack(side="left", padx=16)

        self.lbl_alerts = ttk.Label(top, text=" ", font=("Segoe UI", 11))
        self.lbl_alerts.pack(side="left", padx=16)

        # Chart: liquidation notional rolling
        self.fig_over = Figure(figsize=(9, 4), dpi=100)
        self.ax_over = self.fig_over.add_subplot(111)
        self.ax_over.set_title("Liquidations Notional (last 60 min) - Binance+Bybit (+KuCoin proxy)")
        self.canvas_over = FigureCanvasTkAgg(self.fig_over, master=f)
        self.canvas_over.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    def _build_liq(self):
        f = self.tab_liq
        left = ttk.Frame(f)
        left.pack(side="left", fill="both", expand=True, padx=10, pady=8)

        cols = ("time", "ex", "side", "price", "qty", "notional")
        self.tree_liq = ttk.Treeview(left, columns=cols, show="headings", height=18)
        for c, w in zip(cols, (170, 100, 80, 90, 90, 120)):
            self.tree_liq.heading(c, text=c.upper())
            self.tree_liq.column(c, width=w, anchor="w")
        self.tree_liq.pack(fill="both", expand=True)

        right = ttk.Frame(f)
        right.pack(side="left", fill="both", expand=True, padx=10, pady=8)

        self.fig_liq = Figure(figsize=(6, 4), dpi=100)
        self.ax_liq = self.fig_liq.add_subplot(111)
        self.ax_liq.set_title("Liquidations by Price Bin (last 30 min)")
        self.canvas_liq = FigureCanvasTkAgg(self.fig_liq, master=right)
        self.canvas_liq.get_tk_widget().pack(fill="both", expand=True)

    def _build_book(self):
        f = self.tab_book
        top = ttk.Frame(f)
        top.pack(fill="x", padx=10, pady=8)

        self.lbl_liq_meter = ttk.Label(top, text="Liquidity meter: —", font=("Segoe UI", 11))
        self.lbl_liq_meter.pack(side="left")

        self.fig_book = Figure(figsize=(10, 4), dpi=100)
        self.ax_book = self.fig_book.add_subplot(111)
        self.ax_book.set_title("Top-N Depth Notional (Bid vs Ask)")
        self.canvas_book = FigureCanvasTkAgg(self.fig_book, master=f)
        self.canvas_book.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    def _build_macro(self):
        f = self.tab_macro
        self.fig_macro = Figure(figsize=(10, 4), dpi=100)
        self.ax_macro = self.fig_macro.add_subplot(111)
        self.ax_macro.set_title("Open Interest (snapshots) - last 6 hours")
        self.canvas_macro = FigureCanvasTkAgg(self.fig_macro, master=f)
        self.canvas_macro.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        self.macro_text = tk.Text(f, height=7, wrap="word")
        self.macro_text.pack(fill="x", padx=10, pady=(0,10))
        self.macro_text.insert("end", "Macro notes:\n- Rising OI with falling price often signals leverage building (risk of squeeze/flush).\n- Falling OI with falling price can be long unwind.\n- Funding > 0 favors shorts (longs pay), funding < 0 favors longs (shorts pay).\n")
        self.macro_text.configure(state="disabled")

        self.oi_series: Dict[str, List[OIEntry]] = {"Binance":[], "Bybit":[], "KuCoin":[]}

    def _build_status(self):
        f = self.tab_status
        grid = ttk.Frame(f)
        grid.pack(fill="x", padx=10, pady=8)

        def row(r, label, var, w=12):
            ttk.Label(grid, text=label).grid(row=r, column=0, sticky="w", pady=2, padx=(0,6))
            ttk.Entry(grid, textvariable=var, width=w).grid(row=r, column=1, sticky="w", pady=2)

        row(0, "Alert notional (USD):", self.alert_notional, 16)
        row(1, "Alert imbalance (abs):", self.alert_imbalance, 16)
        row(2, "Alert spread (bps):", self.alert_spread_bps, 16)
        row(3, "KuCoin proxy notional:", self.proxy_notional, 16)
        row(4, "KuCoin proxy vol-z:", self.proxy_volz, 16)

        ttk.Label(grid, text="").grid(row=0, column=2, padx=10)
        ttk.Label(grid, text="Tip: if you see 'book full' + low spread + strong imbalance, liquidity is stacked.").grid(row=0, column=3, rowspan=2, sticky="w")

        # status console
        self.txt_status = tk.Text(f, height=20, wrap="none")
        self.txt_status.pack(fill="both", expand=True, padx=10, pady=10)
        self.txt_status.configure(state="disabled")

    # ---------------------------
    # Start/Stop
    # ---------------------------

    def start(self):
        if self._running:
            self.log_status("Already running.")
            return
        self._running = True

        # Connectors
        self.bin = BinanceConnector(self.sym_binance.get().strip().upper(), self.q, self.log_status, book_depth=self.depth_n.get())
        self.byb = BybitConnector(self.sym_bybit.get().strip().upper(), self.q, self.log_status, book_depth=self.depth_n.get())
        self.kuc = KucoinConnector(self.sym_kucoin.get().strip().upper(), self.q, self.log_status, book_depth=self.depth_n.get())

        self.bin.start_ws()
        self.byb.start_ws()
        self.kuc.start_ws()

        # REST polling thread: book + oi/funding + kucoin proxy
        t = threading.Thread(target=self._poll_loop, daemon=True)
        self._threads = [t]
        t.start()

        self.log_status("Started.")

    def stop(self):
        if not self._running:
            self.log_status("Not running.")
            return
        self._running = False
        for c in (self.bin, self.byb, self.kuc):
            try:
                if c:
                    c.stop()
            except Exception:
                pass
        self.log_status("Stop requested (threads will wind down).")

    def on_close(self):
        try:
            self.stop()
        finally:
            self.root.destroy()

    def test_alert(self):
        self._fire_alert("Test alert: everything is wired.", critical=False)

    # ---------------------------
    # Poll loop
    # ---------------------------

    def _poll_loop(self):
        # refresh cadence
        book_s = 3
        oi_s = 20
        ku_proxy_s = 5

        last_book = 0
        last_oi = 0
        last_proxy = 0

        while self._running and not (self.bin.stopped() and self.byb.stopped() and self.kuc.stopped()):
            ts = now_ts()
            try:
                if ts - last_book >= book_s:
                    last_book = ts
                    # Binance
                    b = self.bin.fetch_book()
                    if b:
                        self.q.put(("book", b))
                    # KuCoin
                    k = self.kuc.fetch_book()
                    if k:
                        self.q.put(("book", k))
                    # Bybit fallback if no recent ws book (we still accept ws updates)
                    bb = self.byb.fetch_book_rest()
                    if bb:
                        self.q.put(("book", bb))

                if ts - last_oi >= oi_s:
                    last_oi = ts
                    for conn, name in [(self.bin,"Binance"), (self.byb,"Bybit"), (self.kuc,"KuCoin")]:
                        oi = conn.fetch_oi_funding()
                        if oi:
                            self.q.put(("oi", oi))

                if ts - last_proxy >= ku_proxy_s:
                    last_proxy = ts
                    ev = self.kuc.compute_proxy_flush(window_s=120,
                                                      notional_thresh=float(self.proxy_notional.get()),
                                                      vol_z=float(self.proxy_volz.get()))
                    if ev:
                        self.q.put(("liq", ev))
                        self.q.put(("proxy", ev))
            except Exception:
                self.log_status("Poll loop error:\n" + traceback.format_exc())

            time.sleep(0.25)

    # ---------------------------
    # Event handling / UI updates
    # ---------------------------

    def _drain_queue(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "liq":
                    self._on_liq(payload)
                elif kind == "book":
                    self._on_book(payload)
                elif kind == "oi":
                    self._on_oi(payload)
                elif kind == "ku_trade":
                    # reserved for future UI
                    pass
                elif kind == "proxy":
                    self._on_proxy(payload)
        except queue.Empty:
            pass
        finally:
            self.root.after(250, self._drain_queue)

    def _on_liq(self, ev: LiqEvent):
        self.liq_events.append(ev)
        if len(self.liq_events) > self.max_liq:
            self.liq_events = self.liq_events[-self.max_liq:]

        # Update table (top)
        try:
            self.tree_liq.insert("", 0, values=(
                fmt_ts(ev.ts),
                ev.exchange,
                ev.side,
                f"{ev.price:,.2f}",
                f"{ev.qty:,.4f}",
                f"{ev.notional:,.0f}"
            ))
            # keep tree bounded
            if len(self.tree_liq.get_children()) > 200:
                for iid in self.tree_liq.get_children()[200:]:
                    self.tree_liq.delete(iid)
        except Exception:
            pass

        # Alert
        if self.alert_enabled.get() and ev.notional >= float(self.alert_notional.get()):
            self._fire_alert(f"{ev.exchange} liquidation spike: {ev.side} ~${ev.notional:,.0f} @ {ev.price:,.2f}", critical=True)

    def _on_proxy(self, ev: LiqEvent):
        self.ku_proxy_events.append(ev)
        if len(self.ku_proxy_events) > self.max_proxy:
            self.ku_proxy_events = self.ku_proxy_events[-self.max_proxy:]

    def _on_book(self, snap: BookSnapshot):
        self.book[snap.exchange] = snap

        # Book alerts
        if self.alert_enabled.get():
            if abs(snap.imbalance) >= float(self.alert_imbalance.get()):
                self._fire_alert(f"{snap.exchange} book imbalance: {snap.imbalance:+.2f} (DepthN={self.depth_n.get()})", critical=False)
            if snap.spread_bps >= float(self.alert_spread_bps.get()):
                self._fire_alert(f"{snap.exchange} spread widened: {snap.spread_bps:.1f} bps", critical=False)

        # Overview label
        self._update_overview_labels()

    def _on_oi(self, oi: OIEntry):
        self.oi[oi.exchange] = oi
        if oi.exchange in self.oi_series:
            self.oi_series[oi.exchange].append(oi)
            # keep 6 hours
            cutoff = now_ts() - 6*3600
            self.oi_series[oi.exchange] = [x for x in self.oi_series[oi.exchange] if x.ts >= cutoff]

    def _update_overview_labels(self):
        # pick a "reference price": Binance best mid if available else others
        price = None
        if "Binance" in self.book:
            b = self.book["Binance"]
            price = (b.bid + b.ask)/2
        elif "Bybit" in self.book:
            b = self.book["Bybit"]
            price = (b.bid + b.ask)/2
        elif "KuCoin" in self.book:
            b = self.book["KuCoin"]
            price = (b.bid + b.ask)/2

        if price:
            self.lbl_price.configure(text=f"ETH reference: ${price:,.2f}")

        # liquidity meter: show each exchange depth + imbalance
        parts = []
        for ex in ["Binance","Bybit","KuCoin"]:
            if ex in self.book:
                s = self.book[ex]
                parts.append(f"{ex}: depth(b/a) ${s.bid_depth/1e6:.2f}M/${s.ask_depth/1e6:.2f}M imb {s.imbalance:+.2f} spr {s.spread_bps:.1f}bps")
        self.lbl_book.configure(text=" | ".join(parts) if parts else "Book: —")

        # alerts summary: last big liq time
        if self.liq_events:
            last = self.liq_events[-1]
            self.lbl_alerts.configure(text=f"Last liq: {last.exchange} {last.side} ${last.notional/1e6:.2f}M @ {last.price:,.0f} ({fmt_ts(last.ts)})")
        else:
            self.lbl_alerts.configure(text="Last liq: —")

    def _fire_alert(self, msg: str, critical: bool = False):
        beep()
        self.log_status("ALERT: " + msg)
        # Popup (non-blocking-ish by using after)
        def show():
            try:
                if critical:
                    messagebox.showwarning("ALERT", msg)
                else:
                    messagebox.showinfo("Notice", msg)
            except Exception:
                pass
        # Avoid spamming: only one popup per 2 seconds
        now = now_ts()
        last = getattr(self, "_last_popup", 0.0)
        if now - last >= 2.0:
            setattr(self, "_last_popup", now)
            self.root.after(10, show)

    # ---------------------------
    # Charts
    # ---------------------------

    def _refresh_charts(self):
        try:
            self._plot_overview()
            self._plot_liq_bins()
            self._plot_book_depth()
            self._plot_macro_oi()
        except Exception:
            self.log_status("Chart refresh error:\n" + traceback.format_exc())
        finally:
            self.root.after(1000, self._refresh_charts)

    def _plot_overview(self):
        ax = self.ax_over
        ax.clear()

        # last 60 min notional time series
        cutoff = now_ts() - 3600
        evs = [e for e in self.liq_events if e.ts >= cutoff]
        if not evs:
            ax.set_title("Liquidations Notional (last 60 min) - waiting for data…")
            self.canvas_over.draw()
            return

        # build per-exchange buckets
        xs = np.array([e.ts for e in evs])
        ys = np.array([e.notional for e in evs])

        # We'll plot notional points and a rolling sum curve
        # rolling 5-min sum
        order = np.argsort(xs)
        xs = xs[order]; ys = ys[order]
        roll_x = []
        roll_y = []
        for i in range(len(xs)):
            t = xs[i]
            w = t - 300
            s = float(np.sum(ys[(xs >= w) & (xs <= t)]))
            roll_x.append(t); roll_y.append(s)

        ax.plot([time.strftime("%H:%M", time.localtime(t)) for t in xs], ys, linestyle="none", marker="o", markersize=3)
        ax.plot([time.strftime("%H:%M", time.localtime(t)) for t in roll_x], roll_y, linewidth=1.8)

        ax.set_ylabel("Notional (USD)")
        ax.set_xlabel("Time (local)")
        ax.grid(True, alpha=0.25)
        ax.set_title("Liquidations Notional: dots=events, line=rolling 5m sum")

        self.canvas_over.draw()

    def _plot_liq_bins(self):
        ax = self.ax_liq
        ax.clear()
        cutoff = now_ts() - 1800
        evs = [e for e in self.liq_events if e.ts >= cutoff]
        if not evs:
            ax.set_title("Liquidations by Price Bin (last 30 min) - waiting for data…")
            self.canvas_liq.draw()
            return

        prices = np.array([e.price for e in evs], dtype=float)
        notionals = np.array([e.notional for e in evs], dtype=float)
        if len(prices) < 2:
            self.canvas_liq.draw()
            return

        # bin around current price
        pmin, pmax = float(np.min(prices)), float(np.max(prices))
        bins = max(10, min(30, int((pmax - pmin)/5) if (pmax-pmin) > 0 else 10))
        hist, edges = np.histogram(prices, bins=bins, weights=notionals)
        centers = 0.5*(edges[:-1] + edges[1:])

        ax.barh(centers, hist, height=(edges[1]-edges[0])*0.9)
        ax.set_xlabel("Notional (USD)")
        ax.set_ylabel("Price")
        ax.grid(True, axis="x", alpha=0.25)
        ax.set_title("Liquidations Notional by Price Bin (last 30 min)")

        self.canvas_liq.draw()

    def _plot_book_depth(self):
        ax = self.ax_book
        ax.clear()

        exs = ["Binance","Bybit","KuCoin"]
        bid = []
        ask = []
        labels = []
        imb = []
        spr = []
        for ex in exs:
            if ex in self.book:
                s = self.book[ex]
                labels.append(ex)
                bid.append(s.bid_depth)
                ask.append(s.ask_depth)
                imb.append(s.imbalance)
                spr.append(s.spread_bps)

        if not labels:
            ax.set_title("Top-N Depth Notional (Bid vs Ask) - waiting for data…")
            self.canvas_book.draw()
            return

        x = np.arange(len(labels))
        ax.bar(x-0.15, bid, width=0.3, label="Bid depth")
        ax.bar(x+0.15, ask, width=0.3, label="Ask depth")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Depth Notional (USD) over top-N levels")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()

        # meter text
        meter = []
        for i, ex in enumerate(labels):
            meter.append(f"{ex} imb {imb[i]:+.2f} spr {spr[i]:.1f}bps")
        self.lbl_liq_meter.configure(text="Liquidity meter: " + " | ".join(meter))

        self.canvas_book.draw()

    def _plot_macro_oi(self):
        ax = self.ax_macro
        ax.clear()

        any_data = False
        for ex, series in self.oi_series.items():
            if not series:
                continue
            any_data = True
            xs = [s.ts for s in series]
            ys = [s.open_interest for s in series]
            ax.plot([time.strftime("%H:%M", time.localtime(t)) for t in xs], ys, linewidth=1.6, label=ex)

        if not any_data:
            ax.set_title("Open Interest (snapshots) - waiting for data…")
            self.canvas_macro.draw()
            return

        ax.set_ylabel("Open Interest (contract units / exchange-defined)")
        ax.set_xlabel("Time (local)")
        ax.grid(True, alpha=0.25)
        ax.legend()

        # Also show latest funding in title line
        fr_parts = []
        for ex in ["Binance","Bybit","KuCoin"]:
            if ex in self.oi and self.oi[ex].funding_rate is not None:
                fr_parts.append(f"{ex} funding {self.oi[ex].funding_rate:+.6f}")
        if fr_parts:
            ax.set_title("Open Interest (last 6h) | " + " | ".join(fr_parts))
        else:
            ax.set_title("Open Interest (last 6h)")

        self.canvas_macro.draw()

    # ---------------------------
    # Run
    # ---------------------------

    def run(self):
        self.log_status("Ready. Click Start.")
        self.root.mainloop()


def main():
    app = App()
    app.run()

if __name__ == "__main__":
    main()
