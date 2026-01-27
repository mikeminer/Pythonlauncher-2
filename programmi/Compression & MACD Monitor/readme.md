# 🧠 ETH/USDT Compression & MACD Monitor

### KuCoin – Windows Desktop Tool

Desktop tool in **Python (Tkinter)** per monitorare **compressioni di mercato**, comportamento del **MACD** e fasi di **contenimento o rilascio del prezzo**.

Progettato per trader discrezionali che vogliono **capire cosa sta accadendo dietro al movimento del prezzo**, non limitarsi a leggere indicatori isolati.

---

## 🎯 Obiettivo del progetto

Questo tool non nasce per “prevedere il mercato”.

Nasce per **interpretare il comportamento strutturale del prezzo**, in particolare nei momenti in cui:

* il mercato resta compresso
* il momentum migliora ma il prezzo non segue
* i breakout vengono continuamente rimandati
* sembra che “qualcuno stia guadagnando tempo”

L’obiettivo è fornire **consapevolezza**, non segnali automatici.

---

## 🚀 Funzionalità principali

✅ Dashboard desktop Windows
✅ Interfaccia grafica Tkinter (nessun browser richiesto)
✅ Grafico prezzo con **Bollinger Bands**
✅ Grafico **MACD + Histogram**
✅ Stream del prezzo in tempo reale (via dati pubblici KuCoin)

### Rilevamento automatico di:

* 🔒 **Compressione di volatilità**
* 🧲 **Containment (momentum trattenuto)**
* 🚀 **Release (possibile rilascio del prezzo)**

### Sistema di alert:

* 🔊 suoni Windows
* 🪟 popup descrittivi
* 🧾 log eventi in tempo reale
* ⏱️ cooldown automatico anti-spam

✅ Nessuna API key richiesta
✅ Utilizza esclusivamente endpoint pubblici KuCoin

---

## 🎯 Mercato supportato

**Default**

* Exchange: **KuCoin**
* Pair: **ETH/USDT**
* Timeframe: **15 minuti**

Tutti i parametri sono **modificabili dall’interfaccia**.

---

## 🧩 Indicatori utilizzati

### 🔹 Bollinger Bands

* Upper Band
* Middle Band (MB)
* Lower Band
* Bollinger Width (ampiezza)

La Bollinger Width è utilizzata per valutare **la compressione della volatilità**, non per segnali di breakout diretti.

---

### 🔹 Compressione di volatilità

La compressione viene calcolata tramite:

* analisi della **Bollinger Width**
* confronto con il **percentile storico** su finestra mobile

Quando la BB Width si trova nei percentili più bassi, il mercato viene classificato come:

🔒 **fase di compressione**

Questo approccio consente di evitare soglie statiche arbitrarie, adattando la lettura al comportamento storico del mercato.

---

### 🔹 MACD

Componenti utilizzati:

* MACD line
* Signal line
* Histogram

Il MACD **non viene usato come segnale long/short**, ma come strumento di lettura del momentum interno.

Serve a comprendere **se il momentum sta cambiando**, anche quando il prezzo non lo riflette ancora.

---

## 🧠 Logica di mercato (parte centrale del tool)

Questo strumento **non dice**:

> “compra” o “vendi”.

Serve a capire **cosa stanno facendo i market maker**.

---

### 🧲 Containment Flag

Si attiva quando:

* il MACD histogram **inizia a risalire**
* ma resta **ancora sotto lo zero**
* il prezzo rimane **vicino alla middle band**

Interpretazione:

> Il momentum tende a migliorare,
> ma il prezzo viene temporaneamente contenuto.

Comportamento tipico di:

* accumulo mascherato
* gestione del tempo
* riduzione del rischio direzionale
* controllo della volatilità

---

### 🚀 Release Flag

Si attiva quando:

* la Bollinger Width smette di contrarsi
* inizia la **prima riespansione**
* il prezzo **chiude sopra la middle band**

Interpretazione:

> Possibile rilascio della compressione
> e inizio di movimento direzionale.

Non indica direzione certa, ma **transizione di regime**.

---

## 🔔 Alert disponibili

Ogni alert genera:

* 🔊 suono Windows
* 🪟 popup descrittivo
* 🧾 log interno

Alert configurabili:

* Breakout sopra Upper Bollinger
* MACD Histogram > 0
* Containment Flag
* Release Flag

Tutti gli alert includono **cooldown automatico** per evitare notifiche ripetitive.

---

## 🖥️ Interfaccia

### Pannello sinistro (scrollabile)

* Exchange
* Pair
* Timeframe
* Numero candele
* Refresh in secondi
* Parametri indicatori
* Soglie di compressione
* Attivazione alert
* Pulsanti Start / Stop

> ⚠️ Se non vedi tutti i controlli, usa la rotellina del mouse.
> Il pannello è completamente scrollabile.

---

### Pannello destro

* Grafico prezzo + Bollinger Bands
* Grafico MACD
* Tabella ultime candele
* Log eventi in tempo reale

---

## ❌ Cosa questo tool NON fa

* Non è un bot di trading
* Non apre né chiude posizioni
* Non fornisce segnali finanziari
* Non predice il futuro
* Non garantisce movimenti di prezzo

È uno **strumento di lettura strutturale del mercato**.

---

## 🧠 Filosofia del progetto

I mercati non si muovono solo per indicatori.

Si muovono per:

* gestione del rischio
* gestione della liquidità
* gestione del tempo

Questo tool nasce dall’osservazione ripetuta di fasi in cui:

* il momentum cambia
* ma il prezzo viene temporaneamente trattenuto

L’obiettivo non è anticipare il mercato,
ma **comprendere il comportamento interno del prezzo**.

---

## 🧪 Requisiti

* Windows 10 / 11
* Python 3.10+
* Connessione Internet

---

## ▶️ Avvio

```bash
pip install -r requirements.txt
python main.py
```

Oppure utilizzare `run.bat`.

---

## ⚠️ Disclaimer

Questo progetto è fornito esclusivamente a scopo educativo e di analisi del mercato.
Non costituisce consulenza finanziaria.

L’uso è a totale responsabilità dell’utente.

