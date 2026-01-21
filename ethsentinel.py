
# ======================================================
# ETH SENTINEL — Python 3.13 compatible (NO pandas)
# Windows alarm + Telegram alert
# ======================================================

import time
import threading
import requests
import numpy as np
import ccxt
import winsound
import tkinter as tk

# ================= CONFIG =================

EXCHANGE = "binance"
SYMBOL = "ETH/USDT"
TIMEFRAME = "1m"
LOOKBACK = 300
POLL_SECONDS = 20

TELEGRAM_TOKEN = "PUT_YOUR_BOT_TOKEN"
TELEGRAM_CHAT_ID = "PUT_YOUR_CHAT_ID"

ALARM_FREQ = 880
ALARM_MS = 400
ALARM_GAP = 150

COOLDOWN = 300

# RSI / MACD / BB
RSI_LEN = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_LEN = 20
BB_STD = 2.0

# ================= TELEGRAM =================

def telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=10)

# ================= INDICATORS =================

def ema(arr, length):
    alpha = 2 / (length + 1)
    out = [arr[0]]
    for price in arr[1:]:
        out.append(alpha * price + (1 - alpha) * out[-1])
    return np.array(out)

def rsi(closes, length):
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains[:length])
    avg_loss = np.mean(losses[:length])

    for i in range(length, len(deltas)):
        avg_gain = (avg_gain * (length - 1) + gains[i]) / length
        avg_loss = (avg_loss * (length - 1) + losses[i]) / length

    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def macd(closes):
    fast = ema(closes, MACD_FAST)
    slow = ema(closes, MACD_SLOW)
    macd_line = fast[-len(slow):] - slow
    signal = ema(macd_line, MACD_SIGNAL)
    return macd_line[-1], signal[-1], macd_line[-2], signal[-2]

def bollinger(closes):
    window = closes[-BB_LEN:]
    mean = np.mean(window)
    std = np.std(window)
    return mean + BB_STD * std, mean - BB_STD * std

# ================= ALARM =================

def alarm_loop(stop):
    while not stop.is_set():
        winsound.Beep(ALARM_FREQ, ALARM_MS)
        time.sleep(ALARM_GAP / 1000)

def alert_window(msg):
    stop = threading.Event()
    t = threading.Thread(target=alarm_loop, args=(stop,), daemon=True)
    t.start()

    root = tk.Tk()
    root.title("🚨 ETH SENTINEL ALERT 🚨")
    root.attributes("-topmost", True)

    tk.Label(root, text=msg, font=("Segoe UI", 12)).pack(padx=20, pady=20)

    def close():
        stop.set()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)
    tk.Button(root, text="Chiudi", command=close).pack(pady=10)
    root.mainloop()

# ================= MAIN =================

def main():
    ex = getattr(ccxt, EXCHANGE)({"enableRateLimit": True})
    last_alert = 0

    print("ETH Sentinel Python 3.13 — avviato")

    while True:
        try:
            candles = ex.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=LOOKBACK)
            closes = np.array([c[4] for c in candles])
            lows = np.array([c[3] for c in candles])

            rsi_val = rsi(closes, RSI_LEN)
            macd_now, sig_now, macd_prev, sig_prev = macd(closes)
            bb_up, bb_low = bollinger(closes)

            now = time.time()
            signal = False
            reason = ""

            if lows[-1] <= bb_low and rsi_val < 35:
                signal = True
                reason = "BB lower + RSI oversold"

            if macd_prev <= sig_prev and macd_now > sig_now:
                signal = True
                reason = "MACD cross UP"

            if signal and now - last_alert > COOLDOWN:
                msg = f"ETH SENTINEL\n{reason}\nRSI={round(rsi_val,2)}"
                telegram(msg)
                alert_window(msg)
                last_alert = time.time()

        except Exception as e:
            print("ERROR:", e)

        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
