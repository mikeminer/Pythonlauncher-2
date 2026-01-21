<img width="623" height="448" alt="image" src="https://github.com/user-attachments/assets/6b68837c-4b3c-4f0e-b5bd-420291cd7cc8" />

# 🚀 Trading Tools – Python Launcher & Crypto Analytics

Suite di **strumenti Python per analisi crypto, trading realtime e monitoraggio market maker**, con **launcher modulare per Windows**.

Il progetto nasce per gestire in modo ordinato più script Python (radar, reversal, strike, monitor websocket, dashboard... e anche script python personalizzati) utilizzando **un unico ambiente Python controllato**.

---

## 🧠 Funzionalità principali

### ✅ Launcher Windows (`.bat`)

* Menu interattivo
* Selezione versione Python (es. `py -3.11`)
* Salvataggio default persistente
* Avvio multiplo script Python
* Installazione librerie:

  * BASE
  * OPZIONALI
  * Tutto
  * `requirements.txt`
* Apertura rapida strumenti esterni (Coinglass)

### ✅ Tool di trading inclusi

* Radar Spinta Market Maker
* Reversal detector
* Strike / breakout monitor
* WebSocket realtime price feed
* Analisi liquidazioni
* Supporto timeframe 5m / 15m / 30m / 1h
* Notifiche Windows
* Dashboard GUI

---

## 📊 Integrazione esterna

Dal menu è possibile aprire direttamente:

🔗 **Coinglass – Liquidation Heatmap**
[https://www.coinglass.com/liquidation-levels](https://www.coinglass.com/liquidation-levels)

Utilizzata come supporto visivo per:

* zone di liquidità
* cluster di leva
* livelli di possibile squeeze

---

## 🧩 Stack tecnologico

### Python consigliato

* **Python 3.11.x (fortemente consigliato)**
* Compatibile Windows 10 / 11

> ⚠️ Python 3.12 è sconsigliato per instabilità con NumPy / Matplotlib.

---

## 📦 Librerie utilizzate

### Base

* `requests`
* `websockets`
* `websocket-client`
* `ccxt`
* `numpy`
* `tzdata`

### Opzionali (dashboard / analisi)

* `pandas`
* `matplotlib`
* `mplfinance`
* `plyer`
* `psutil`

---

## 📄 requirements.txt (consigliato)

```txt
requests>=2.31.0
websockets>=12.0
websocket-client>=1.7.0
ccxt>=4.0.0
numpy>=1.26.0
pandas>=2.2.0
matplotlib>=3.8.0
mplfinance>=0.12.10b0
plyer>=2.1.0
psutil>=5.9.8
tzdata>=2024.1
```

Installazione:

```bash
py -3.11 -m pip install -r requirements.txt
```

---

## ▶️ Avvio

### Metodo consigliato

Avviare il file:

```
Launcher_TradingTools_Final.bat
```

Da qui è possibile:

* scegliere Python
* installare librerie
* avviare gli script
* aprire strumenti esterni
* gestire l’ambiente

---

## 📁 Struttura progetto

```
trading-tools/
│
├─ Launcher_TradingTools_Final.bat
├─ python_default.cmd
├─ requirements.txt
│
├─ radar.py
├─ reversal.py
├─ strike.py
├─ websocket_feed.py
│
└─ README.md
```

---

## ⚠️ Disclaimer

Questo progetto è a scopo:

* educativo
* sperimentale
* di studio dei mercati crypto

❗ Non costituisce consulenza finanziaria.
L’utilizzo è sotto responsabilità dell’utente.

---

## 🧠 Filosofia del progetto

> “Non cercare di prevedere il mercato.
> Cerca dove è concentrata la liquidità.”

Il sistema è progettato per:

* leggere il comportamento del prezzo
* individuare zone di accumulo
* riconoscere transizioni da accumulo → spinta
* evitare liquidazioni inutili

---

## 🛠️ Stato del progetto

* ✔ funzionante
* ✔ modulare
* ✔ espandibile
* ✔ compatibile con nuovi tool

Progetto in continua evoluzione.

---

## 📌 Roadmap (future idee)

* Dashboard unica integrata
* Aggregazione liquidazioni multi-exchange
* Alert automatici su shift market maker
* Pattern detection AI-assisted
* Export segnali
* Versione standalone (.exe)

---

## 🤝 Contributi

Pull request, idee e miglioramenti sono benvenuti.
pullshark
pair
pair-2
shark2
shark2
shark2
shark2
shark2
