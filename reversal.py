# ======================================================
# ETH SENTINEL — Python 3.13 compatible (NO pandas)
# Telegram alert + Windows continuous alarm until window closed
# Improved: timestamps + stronger signal filters to reduce false MACD crosses
# ======================================================

import time
import threading
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests
import numpy as np
import ccxt
import winsound
import tkinter as tk

# ================= CONFIG =================

EXCHANGE = "binance"
SYMBOL = "ETH/USDT"
TIMEFRAME = "1m"
LOOKBACK = 400          # a bit more history for EMA filters
POLL_SECONDS = 20

TELEGRAM_TOKEN = "PUT_YOUR_BOT_TOKEN"
TELEGRAM_CHAT_ID = "PUT_YOUR_CHAT_ID"

ALARM_FREQ = 880
ALARM_MS = 400
ALARM_GAP = 150

COOLDOWN = 300          # seconds

# Indicators
RSI_LEN = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_LEN = 20
BB_STD = 2.0

# Filters to reduce spam / whipsaws
EMA_TREND_LEN = 50              # trend filter
MACD_EPS = 0.20                 # minimal separation (tune by market; units ~ $ depends on price scale)
CONFIRM_BARS = 1                # require 1 extra closed candle confirmation

# Oversold logic (less noisy)
RSI_OVERSOLD = 35
RSI_TURNUP_LEVEL = 30           # require RSI to cross UP this level for BB event

TZ_LOCAL = ZoneInfo("Europe/Rome")

# ================= UTIL =================

def ts_strings():
    now_utc = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(TZ_LOCAL)
    return now_local.strftime("%Y-%m-%d %H:%M:%S"), now_utc.strftime("%Y-%m-%d %H:%M:%S")

def telegram(msg: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    r = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=10)
    r.raise_for_status()

# ================= INDICATORS (numpy) =================

def ema(arr: np.ndarray, length: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    alpha = 2.0 / (length + 1.0)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out

def rsi_series(closes: np.ndarray, length: int) -> np.ndarray:
    closes = np.asarray(closes, dtype=float)
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    rsi_vals = np.full(closes.shape, np.nan, dtype=float)
    if len(deltas) < length + 2:
        return rsi_vals

    avg_gain = np.mean(gains[:length])
    avg_loss = np.mean(losses[:length])

    # first RSI index corresponds to closes[length]
    rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
    rsi_vals[length] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(length, len(deltas)):
        avg_gain = (avg_gain * (length - 1) + gains[i]) / length
        avg_loss = (avg_loss * (length - 1) + losses[i]) / length
        rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
        rsi_vals[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    return rsi_vals

def macd_series(closes: np.ndarray, fast: int, slow: int, signal: int):
    closes = np.asarray(closes, dtype=float)
    efast = ema(closes, fast)
    eslow = ema(closes, slow)
    macd_line = efast - eslow
    sig_line = ema(macd_line, signal)
    hist = macd_line - sig_line
    return macd_line, sig_line, hist

def bollinger_bands(closes: np.ndarray, length: int, std_mult: float):
    closes = np.asarray(closes, dtype=float)
    if len(closes) < length:
        return np.nan, np.nan, np.nan
    window = closes[-length:]
    mid = float(np.mean(window))
    std = float(np.std(window, ddof=0))
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return upper, mid, lower

# ================= ALARM + WINDOW =================

def alarm_loop(stop_event: threading.Event):
    while not stop_event.is_set():
        winsound.Beep(ALARM_FREQ, ALARM_MS)
        time.sleep(ALARM_GAP / 1000.0)

def alert_window(msg: str):
    stop = threading.Event()
    t = threading.Thread(target=alarm_loop, args=(stop,), daemon=True)
    t.start()

    root = tk.Tk()
    root.title("🚨 ETH SENTINEL ALERT 🚨")
    root.attributes("-topmost", True)

    tk.Label(root, text=msg, font=("Segoe UI", 11), justify="left").pack(padx=16, pady=16)

    def close():
        stop.set()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)
    tk.Button(root, text="Chiudi e silenzia", command=close).pack(pady=(0, 14))
    root.mainloop()

# ================= SIGNAL LOGIC =================

def crossed_up(prev: float, cur: float) -> bool:
    return prev <= 0.0 and cur > 0.0

def main():
    ex = getattr(ccxt, EXCHANGE)({"enableRateLimit": True})
    last_alert = 0.0

    print("ETH Sentinel Python 3.13 — avviato")

    while True:
        try:
            candles = ex.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=LOOKBACK)
            # Use only closed candles: ccxt returns recent; last candle may be still forming.
            # We'll treat last as "latest" but require confirmation bars to reduce false triggers.
            closes = np.array([c[4] for c in candles], dtype=float)
            lows = np.array([c[3] for c in candles], dtype=float)

            # Indicators
            rsi_vals = rsi_series(closes, RSI_LEN)
            macd_line, sig_line, hist = macd_series(closes, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
            bb_up, bb_mid, bb_low = bollinger_bands(closes, BB_LEN, BB_STD)
            ema_trend = ema(closes, EMA_TREND_LEN)

            # Use indices for confirmation
            # We confirm signals using the last (CONFIRM_BARS+1) closed candles.
            # Example CONFIRM_BARS=1: signal must occur at i=-2 and still valid at i=-1.
            i_signal = -2 - (CONFIRM_BARS - 0)   # where crossing happened
            i_confirm = -1                      # latest close

            # Safety for small arrays
            if len(closes) < 60:
                time.sleep(POLL_SECONDS)
                continue

            # Trend filter: only consider bullish-type alerts if close > EMA(50) or EMA slope up
            ema_slope = ema_trend[-1] - ema_trend[-5]
            trend_ok = (closes[-1] > ema_trend[-1]) or (ema_slope > 0)

            now = time.time()
            if now - last_alert < COOLDOWN:
                time.sleep(POLL_SECONDS)
                continue

            local_ts, utc_ts = ts_strings()

            # -------------------------
            # EVENT A: BB lower touch + RSI turn up (less noisy)
            # Condition:
            # - low touches/breaks lower BB on the signal candle
            # - RSI was oversold and crosses UP RSI_TURNUP_LEVEL between prev and last confirmed
            # -------------------------
            bb_touch = lows[i_confirm] <= bb_low

            rsi_prev = rsi_vals[-2]
            rsi_now = rsi_vals[-1]
            rsi_turn_up = (not np.isnan(rsi_prev)) and (not np.isnan(rsi_now)) and (rsi_prev < RSI_TURNUP_LEVEL <= rsi_now)

            bb_rsi_event = bb_touch and rsi_turn_up

            # -------------------------
            # EVENT B: MACD cross up with confirmation + epsilon + trend filter
            # - histogram crosses up through 0 on signal candle
            # - and stays positive on confirm candle
            # - and macd-signal separation >= MACD_EPS on confirm candle
            # - and trend_ok
            # -------------------------
            hist_prev = hist[i_signal - 1]
            hist_sig = hist[i_signal]
            hist_conf = hist[i_confirm]

            macd_sep = float(macd_line[i_confirm] - sig_line[i_confirm])
            macd_event = (
                crossed_up(float(hist_prev), float(hist_sig)) and
                float(hist_conf) > 0.0 and
                macd_sep >= MACD_EPS and
                trend_ok
            )

            # Decide
            if bb_rsi_event or macd_event:
                if bb_rsi_event:
                    reason = "BB lower touch + RSI turn-up"
                else:
                    reason = "MACD cross UP (confirmed + filtered)"

                msg = (
                    f"ETH SENTINEL\n"
                    f"{reason}\n"
                    f"Local: {local_ts} (Rome)\n"
                    f"UTC:   {utc_ts}\n"
                    f"Close: {closes[-1]:.2f}\n"
                    f"RSI:   {rsi_now:.2f}\n"
                    f"MACDΔ: {macd_sep:.4f}\n"
                    f"Trend: {'OK' if trend_ok else 'NO'}"
                )

                telegram(msg)
                alert_window(msg)
                last_alert = time.time()

        except Exception as e:
            print("ERROR:", e)

        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
