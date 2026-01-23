# 🛡️ Guardiano Anti-Martingala — ETH

### Disciplina prima del profitto

> Un software in Python con interfaccia grafica che impedisce al trader di distruggere il conto mediando in modo emotivo.
<img width="1307" height="767" alt="image" src="https://github.com/user-attachments/assets/37767f2d-9b30-47d4-81cd-6b69dafdbb34" />

---

## 🚀 Cos’è

**Guardiano Anti-Martingala** è un tool disciplinare progettato per:

* ❌ bloccare il martingala emotivo
* ✅ permettere aggiunte **solo se le regole sono rispettate**
* 🧠 trasformare il trading da impulso a processo
* 📊 rendere visibile il rischio reale prima di ogni decisione

Non è un bot.
Non entra a mercato.
Non dà segnali.

👉 È un **filtro di sopravvivenza**.

---

## 🎯 Perché esiste

La maggior parte dei conti non muore per un trade sbagliato, ma per questo:

> “Scende ancora… aggiungo.”

Questo software risponde con:

> “Aspetta. Vediamo se puoi davvero farlo.”

---

## ⚙️ Funzionalità principali

### ✅ Dashboard operativa

Inserisci:

* equity
* prezzo medio
* size
* prezzo attuale
* stop
* liquidation price
* leva

Il sistema calcola automaticamente:

* distanza dalla liquidation
* rischio su stop (% equity)
* numero di aggiunte già fatte
* distanza dall’ultima media

---

### 🧱 Regole anti-martingala personalizzabili

Puoi impostare:

* massimo numero di aggiunte
* cooldown minimo tra un’aggiunta e l’altra
* distanza minima tra prezzi
* rischio massimo accettabile sull’equity
* distanza minima dalla liquidation
* obbligo di conferme tecniche:

  * sweep + reclaim
  * break & retest
  * spike di volume

Le regole diventano **più forti delle emozioni**.

---

### 🚫 Blocco automatico con spiegazione

Se una richiesta di aggiunta viene rifiutata, il software mostra:

* ❌ motivo tecnico del rifiuto
* 🧠 spiegazione chiara in italiano
* 🧭 cosa fare per tornare nei parametri

Esempio:

> “Stai mediando troppo vicino alla liquidation.
> Questa aggiunta riduce la sopravvivenza del trade.”

---

### 📓 Diario di trading integrato

Ogni evento viene registrato:

* CONSENTITO
* BLOCCATO
* NOTE personali

Il log può essere esportato in **CSV** per analisi futura.

---

### 📡 Prezzo ETH in tempo reale (opzionale)

* feed live da Binance
* aggiornamento automatico
* nessuna chiave API richiesta

---

## 🖥️ Requisiti

* Windows
* Python **3.10 o superiore**
* Librerie:

  * `requests`

---

## 📦 Installazione

```bash
git clone https://github.com/mikeminer/Guardiano-Anti-Martingala.git
cd Guardiano-Anti-Martingala
python -m pip install -r requirements.txt
python guardiano_anti_martingala.py
```

---

## 🧠 Filosofia

> Non si media un’idea sbagliata.
> Si media solo un’idea giusta nel momento sbagliato.

Il Guardiano non ti fa guadagnare di più.
Ti impedisce di perdere **tutto**.

---

## ⚠️ Disclaimer

Questo software:

* ❌ non è consulenza finanziaria
* ❌ non esegue ordini
* ❌ non garantisce profitti

È uno **strumento di disciplina personale**.

Usalo per proteggerti da te stesso.

---

## 👤 Autore

**Michele Angelo Forlani**
alias **Forlani Bank**

> Strategia, disciplina e sopravvivenza prima del profitto.

---

## ⭐ Contributi

Pull request, idee e miglioramenti sono benvenuti.

Se questo progetto ti ha aiutato:

* ⭐ metti una stella
* 🧠 usalo con disciplina
* 🔒 proteggi il capitale

