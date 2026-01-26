
# 🧠 ETH/USDT Compression & MACD Monitor

**KuCoin – Windows Desktop Tool**

Tool desktop in **Python (Tkinter)** per monitorare le fasi di **compressione di mercato**, il comportamento del **MACD** e individuare i momenti in cui i market maker **contengono** o **rilasciano** il prezzo.

Pensato per trader discrezionali che vogliono **capire cosa sta succedendo dietro al movimento**, non solo vedere indicatori.

---

## 🚀 Funzionalità principali

✅ Dashboard **desktop Windows** 
✅ Grafico **Prezzo + Bollinger Bands**
✅ Grafico **MACD + Histogram**
✅ Rilevamento automatico di:

* **Compressione di volatilità**
* **Containment (MACD tenuto negativo artificialmente)**
* **Release (possibile rilascio del prezzo)**

✅ **Alert sonori Windows**
✅ **Popup descrittivi**
✅ Log eventi in tempo reale
✅ Aggiornamento automatico ogni N secondi
✅ Nessuna API key richiesta (usa OHLCV pubblici KuCoin)

---

## 🎯 Mercato supportato

**Default**

* Exchange: **KuCoin**
* Pair: **ETH/USDT**
* Timeframe: **15m**

Modificabili liberamente dall’interfaccia.

---

## 🧩 Indicatori utilizzati

### 🔹 Bollinger Bands

* Upper Band
* Middle Band (MB)
* Lower Band
* **BB Width** (ampiezza)

### 🔹 Compressione

La compressione viene calcolata tramite:

* Percentile della Bollinger Width su finestra storica

Quando la BB Width è nei **percentili più bassi**, il mercato è considerato in:

> 🔒 **Compressione di volatilità**

---

### 🔹 MACD

* Linea MACD
* Signal line
* Histogram

Usato non come “segnale long/short”, ma come **strumento di lettura del controllo del momentum**.

---

## 🧠 Logica di mercato (parte importante)

Questo tool NON dice:

> “compra” o “vendi”.

Serve a capire **cosa stanno facendo i market maker**.

---

### 🧲 Containment Flag

Si attiva quando:

* MACD histogram **sta risalendo**
* ma resta **ancora sotto lo zero**
* prezzo resta **vicino alla media Bollinger**

Interpretazione:

> Il momentum vorrebbe girare positivo
> ma il prezzo viene **tenuto sotto controllo**

Tipico comportamento di:

* contenimento
* accumulo mascherato
* gestione del tempo

---

### 🚀 Release Flag

Si attiva quando:

* Bollinger Width inizia a **riespandersi**
* il prezzo chiude **sopra la middle band**

Interpretazione:

> Possibile rilascio della compressione
> inizio movimento direzionale

---

## 🔔 Alert disponibili

Ogni alert genera:

* 🔊 suono Windows
* 🪟 popup descrittivo
* 🧾 log interno

Alert configurabili:

* **Breakout sopra Upper Bollinger**
* **MACD histogram > 0**
* **Release flag**
* **Containment flag**

Gli alert hanno **cooldown automatico** per evitare spam.

---

## 🖥️ Interfaccia

### Pannello sinistro (scrollabile)

* Exchange
* Pair
* Timeframe
* Numero candele
* Refresh secondi
* Parametri indicatori
* Soglie compressione
* Attivazione alert
* Start / Stop

> ⚠️ Se non vedi tutto: usa la **rotellina del mouse**
> Il pannello è scrollabile.

---

### Pannello destro

* Grafico prezzo + Bollinger
* Grafico MACD
* Tabella ultime 20 candele
* Log eventi

---

## 🧪 Requisiti

* Windows 10 / 11
* Python **3.10+**
* Connessione Internet

---

## 📦 Installazione

```powershell
pip install -r requirements.txt
```

---

## ▶ Avvio

```powershell
python main.py
```

Oppure doppio click su:

```
run.bat
```

---

## 🔐 Sicurezza

* Nessuna API key
* Nessun trading automatico
* Nessuna operazione su account
* Solo dati pubblici OHLCV

Tool **100% osservativo**.

---

## ⚠️ Disclaimer

Questo software:

* **non fornisce segnali finanziari**
* **non è un bot di trading**
* **non garantisce risultati**

È uno strumento di **lettura strutturale del mercato**, pensato per supportare il ragionamento del trader.

---

## 🧠 Filosofia del tool

> “Il prezzo mente spesso.
> La volatilità e il tempo mentono molto meno.”

Questo strumento nasce per osservare:

* quando il mercato **non può scendere**
* quando **non vuole ancora salire**
* quando sta **comprando tempo**

---

## 📌 Roadmap (facoltativa)

* [ ] modalità multi-timeframe
* [ ] alert breakout + retest
* [ ] profili di mercato (London / NY)
* [ ] export log
* [ ] versione .exe standalone

