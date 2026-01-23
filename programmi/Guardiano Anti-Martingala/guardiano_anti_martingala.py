# -*- coding: utf-8 -*-
"""
Guardiano Anti-Martingala — ETH (Forlani Bank)
Python 3.10+ | GUI Tkinter | Windows friendly

Cosa fa:
- Ti impedisce di "mediare" (martingala) senza rispettare regole OGGETTIVE.
- Quando una richiesta di aggiunta viene rifiutata, ti spiega *perché* e cosa fare per rientrare nei parametri.
- Log/diario interno + export CSV.
- Prezzo live opzionale via Binance (richiede requests).

Nota: non è consulenza finanziaria. È un "parafango" di disciplina.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import time
import csv
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    import requests
except Exception:
    requests = None

APP_TITLE = "Guardiano Anti-Martingala — ETH (Forlani Bank)"
DEFAULT_SYMBOL = "ETHUSDT"


def pct(a: float, b: float) -> float:
    return 0.0 if a == 0 else (b - a) / a


def safe_float(x: str, default: float = 0.0) -> float:
    try:
        x = str(x).strip().replace(",", ".")
        return default if x == "" else float(x)
    except Exception:
        return default


@dataclass
class Rules:
    max_aggiunte: int = 2
    cooldown_minuti: int = 30
    distanza_min_pct: float = 1.0
    rischio_max_pct_equity: float = 2.0
    margine_min_pct_da_liq: float = 6.0
    richiedi_sweep_reclaim: bool = True
    richiedi_break_retest: bool = False
    richiedi_vol_spike: bool = False


@dataclass
class Position:
    equity_usdt: float = 0.0
    prezzo_medio: float = 0.0
    size_eth: float = 0.0
    prezzo_attuale: float = 0.0
    stop_price: float = 0.0
    liq_price: float = 0.0
    leva: float = 1.0

    aggiunte_effettuate: int = 0
    ultimo_add_ts: float = 0.0
    ultimo_add_price: float = 0.0


class GuardianoApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1060x720")
        self.minsize(980, 640)

        self.rules = Rules()
        self.pos = Position()

        self.price_feed_enabled = tk.BooleanVar(value=True)
        self.symbol_var = tk.StringVar(value=DEFAULT_SYMBOL)
        self.auto_update_secs = tk.IntVar(value=5)

        self._build_ui()

        self.last_price_fetch = 0.0
        self.after(500, self._loop)

    def _build_ui(self):
        try:
            ttk.Style(self).theme_use("clam")
        except Exception:
            pass

        top = ttk.Frame(self, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text=APP_TITLE, font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT)

        right = ttk.Frame(top)
        right.pack(side=tk.RIGHT)

        ttk.Checkbutton(right, text="Prezzo live (Binance)", variable=self.price_feed_enabled).pack(side=tk.LEFT, padx=6)
        ttk.Label(right, text="Simbolo:").pack(side=tk.LEFT)
        ttk.Entry(right, textvariable=self.symbol_var, width=10).pack(side=tk.LEFT, padx=4)
        ttk.Label(right, text="Update (s):").pack(side=tk.LEFT)
        ttk.Spinbox(right, from_=2, to=60, textvariable=self.auto_update_secs, width=5).pack(side=tk.LEFT, padx=4)

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.tab_dashboard = ttk.Frame(self.nb, padding=10)
        self.tab_rules = ttk.Frame(self.nb, padding=10)
        self.tab_log = ttk.Frame(self.nb, padding=10)

        self.nb.add(self.tab_dashboard, text="Dashboard")
        self.nb.add(self.tab_rules, text="Regole")
        self.nb.add(self.tab_log, text="Log / Diario")

        self._build_dashboard_tab()
        self._build_rules_tab()
        self._build_log_tab()

        bottom = ttk.Frame(self, padding=(10, 0, 10, 10))
        bottom.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_var = tk.StringVar(value="Pronto.")
        ttk.Label(bottom, textvariable=self.status_var).pack(side=tk.LEFT)

        ttk.Button(bottom, text="Esporta Log CSV", command=self.export_log).pack(side=tk.RIGHT, padx=6)
        ttk.Button(bottom, text="Reset Sessione", command=self.reset_session).pack(side=tk.RIGHT, padx=6)

    def _build_dashboard_tab(self):
        left = ttk.LabelFrame(self.tab_dashboard, text="Posizione (input)", padding=10)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        self.vars = {}
        fields = [
            ("Equity (USDT)", "equity_usdt"),
            ("Prezzo medio entry", "prezzo_medio"),
            ("Size (ETH)", "size_eth"),
            ("Prezzo attuale", "prezzo_attuale"),
            ("Stop (se lo hai)", "stop_price"),
            ("Liquidation price", "liq_price"),
            ("Leva", "leva"),
        ]
        for i, (label, key) in enumerate(fields):
            ttk.Label(left, text=label).grid(row=i, column=0, sticky="w", pady=5)
            v = tk.StringVar(value="")
            self.vars[key] = v
            ttk.Entry(left, textvariable=v, width=22).grid(row=i, column=1, sticky="w", pady=5)

        ttk.Button(left, text="Applica / Aggiorna", command=self.apply_position).grid(
            row=len(fields), column=0, columnspan=2, pady=(10, 0), sticky="we"
        )

        ttk.Separator(left, orient=tk.HORIZONTAL).grid(row=len(fields) + 1, column=0, columnspan=2, sticky="we", pady=10)

        add_frame = ttk.LabelFrame(left, text="Proponi aggiunta (anti-martingala)", padding=10)
        add_frame.grid(row=len(fields) + 2, column=0, columnspan=2, sticky="we")

        self.add_size_var = tk.StringVar(value="0.5")
        self.add_price_var = tk.StringVar(value="")

        ttk.Label(add_frame, text="Nuova size (ETH)").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(add_frame, textvariable=self.add_size_var, width=18).grid(row=0, column=1, sticky="w", pady=4)
        ttk.Label(add_frame, text="Prezzo aggiunta (vuoto = prezzo attuale)").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(add_frame, textvariable=self.add_price_var, width=18).grid(row=1, column=1, sticky="w", pady=4)

        self.confirm_sweep = tk.BooleanVar(value=False)
        self.confirm_retest = tk.BooleanVar(value=False)
        self.confirm_vol = tk.BooleanVar(value=False)

        ttk.Checkbutton(
            add_frame,
            text="Ho visto sweep+reclaim (wick sotto e chiusura sopra livello)",
            variable=self.confirm_sweep,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=3)
        ttk.Checkbutton(
            add_frame,
            text="Ho visto break+retest (chiusura sopra + pullback e tenuta)",
            variable=self.confirm_retest,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=3)
        ttk.Checkbutton(add_frame, text="Ho visto spike volume", variable=self.confirm_vol).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=3
        )

        ttk.Button(add_frame, text="✅ Valida e registra aggiunta", command=self.propose_add).grid(
            row=5, column=0, columnspan=2, pady=(10, 0), sticky="we"
        )

        right = ttk.LabelFrame(self.tab_dashboard, text="Metriche & Guardrail", padding=10)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.metrics_text = tk.Text(right, height=28, wrap=tk.WORD)
        self.metrics_text.pack(fill=tk.BOTH, expand=True)

        ttk.Button(right, text="Aggiorna metriche", command=self.refresh_metrics).pack(pady=(8, 0), fill=tk.X)

    def _build_rules_tab(self):
        box = ttk.LabelFrame(self.tab_rules, text="Regole personali anti-martingala", padding=10)
        box.pack(fill=tk.BOTH, expand=True)

        self.rule_vars = {
            "max_aggiunte": tk.IntVar(value=self.rules.max_aggiunte),
            "cooldown_minuti": tk.IntVar(value=self.rules.cooldown_minuti),
            "distanza_min_pct": tk.DoubleVar(value=self.rules.distanza_min_pct),
            "rischio_max_pct_equity": tk.DoubleVar(value=self.rules.rischio_max_pct_equity),
            "margine_min_pct_da_liq": tk.DoubleVar(value=self.rules.margine_min_pct_da_liq),
            "richiedi_sweep_reclaim": tk.BooleanVar(value=self.rules.richiedi_sweep_reclaim),
            "richiedi_break_retest": tk.BooleanVar(value=self.rules.richiedi_break_retest),
            "richiedi_vol_spike": tk.BooleanVar(value=self.rules.richiedi_vol_spike),
        }

        rows = [
            ("Max aggiunte consentite", "max_aggiunte"),
            ("Cooldown tra aggiunte (minuti)", "cooldown_minuti"),
            ("Distanza minima tra aggiunte (%)", "distanza_min_pct"),
            ("Rischio max stimato su stop (% equity)", "rischio_max_pct_equity"),
            ("Margine minimo da liquidation (%)", "margine_min_pct_da_liq"),
        ]

        for i, (label, key) in enumerate(rows):
            ttk.Label(box, text=label).grid(row=i, column=0, sticky="w", pady=6)
            var = self.rule_vars[key]
            if isinstance(var, tk.IntVar):
                w = ttk.Spinbox(box, from_=0, to=9999, textvariable=var, width=10)
            else:
                w = ttk.Entry(box, textvariable=var, width=10)
            w.grid(row=i, column=1, sticky="w", pady=6)

        ttk.Separator(box, orient=tk.HORIZONTAL).grid(row=len(rows), column=0, columnspan=2, sticky="we", pady=10)

        ttk.Label(box, text="Conferme obbligatorie (spuntale quando proponi l'aggiunta):", font=("Segoe UI", 10, "bold")).grid(
            row=len(rows) + 1, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )

        ttk.Checkbutton(box, text="Richiedi sweep+reclaim", variable=self.rule_vars["richiedi_sweep_reclaim"]).grid(
            row=len(rows) + 2, column=0, columnspan=2, sticky="w", pady=4
        )
        ttk.Checkbutton(box, text="Richiedi break+retest", variable=self.rule_vars["richiedi_break_retest"]).grid(
            row=len(rows) + 3, column=0, columnspan=2, sticky="w", pady=4
        )
        ttk.Checkbutton(box, text="Richiedi spike volume", variable=self.rule_vars["richiedi_vol_spike"]).grid(
            row=len(rows) + 4, column=0, columnspan=2, sticky="w", pady=4
        )

        ttk.Separator(box, orient=tk.HORIZONTAL).grid(row=len(rows) + 5, column=0, columnspan=2, sticky="we", pady=10)

        btns = ttk.Frame(box)
        btns.grid(row=len(rows) + 6, column=0, columnspan=2, sticky="we")
        ttk.Button(btns, text="Salva regole", command=self.save_rules).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btns, text="Carica regole", command=self.load_rules).pack(side=tk.LEFT)

    def _build_log_tab(self):
        self.log_rows = []
        self.log_tree = ttk.Treeview(self.tab_log, columns=("ts", "azione", "esito", "motivo"), show="headings")
        for col, w in [("ts", 160), ("azione", 210), ("esito", 120), ("motivo", 540)]:
            self.log_tree.heading(col, text=col.upper())
            self.log_tree.column(col, width=w, anchor="w")
        self.log_tree.pack(fill=tk.BOTH, expand=True)

        bar = ttk.Frame(self.tab_log)
        bar.pack(fill=tk.X, pady=8)
        ttk.Button(bar, text="Aggiungi nota al diario", command=self.add_note).pack(side=tk.LEFT)
        ttk.Button(bar, text="Pulisci log", command=self.clear_log).pack(side=tk.LEFT, padx=8)

    def apply_position(self):
        p = self.pos
        p.equity_usdt = safe_float(self.vars["equity_usdt"].get(), 0.0)
        p.prezzo_medio = safe_float(self.vars["prezzo_medio"].get(), 0.0)
        p.size_eth = safe_float(self.vars["size_eth"].get(), 0.0)
        p.prezzo_attuale = safe_float(self.vars["prezzo_attuale"].get(), p.prezzo_attuale)
        p.stop_price = safe_float(self.vars["stop_price"].get(), 0.0)
        p.liq_price = safe_float(self.vars["liq_price"].get(), 0.0)
        p.leva = max(0.01, safe_float(self.vars["leva"].get(), 1.0))
        self.refresh_metrics()
        self.status_var.set("Posizione aggiornata.")

    def refresh_metrics(self):
        p = self.pos
        r = self.rules
        out = []
        out.append(f"Prezzo attuale: {p.prezzo_attuale:,.2f}  |  Prezzo medio: {p.prezzo_medio:,.2f}  |  Size: {p.size_eth:.4f} ETH")
        if p.prezzo_medio > 0 and p.prezzo_attuale > 0:
            out.append(f"Δ da medio: {pct(p.prezzo_medio, p.prezzo_attuale)*100:+.2f}%")
        out.append("")
        if p.liq_price > 0 and p.prezzo_attuale > 0:
            dist_liq = (p.prezzo_attuale - p.liq_price) / p.prezzo_attuale * 100.0
            out.append(f"Distanza da liquidation: {dist_liq:.2f}% (min: {r.margine_min_pct_da_liq:.2f}%)")
        else:
            out.append("Distanza da liquidation: (inserisci liq_price per calcolarla)")
        if p.stop_price > 0 and p.size_eth > 0 and p.equity_usdt > 0:
            risk_usdt = max(0.0, (p.prezzo_medio - p.stop_price) * p.size_eth)
            risk_pct = (risk_usdt / p.equity_usdt) * 100.0
            out.append(f"Rischio stimato su stop: {risk_usdt:,.2f} USDT ({risk_pct:.2f}% equity) (max: {r.rischio_max_pct_equity:.2f}%)")
        else:
            out.append("Rischio su stop: (inserisci equity/stop/size per calcolarlo)")
        out.append("")
        out.append(f"Aggiunte effettuate: {p.aggiunte_effettuate} / {r.max_aggiunte}")
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.insert(tk.END, "\n".join(out))

    def _sync_rules_from_ui(self):
        r = self.rules
        r.max_aggiunte = int(self.rule_vars["max_aggiunte"].get())
        r.cooldown_minuti = int(self.rule_vars["cooldown_minuti"].get())
        r.distanza_min_pct = float(self.rule_vars["distanza_min_pct"].get())
        r.rischio_max_pct_equity = float(self.rule_vars["rischio_max_pct_equity"].get())
        r.margine_min_pct_da_liq = float(self.rule_vars["margine_min_pct_da_liq"].get())
        r.richiedi_sweep_reclaim = bool(self.rule_vars["richiedi_sweep_reclaim"].get())
        r.richiedi_break_retest = bool(self.rule_vars["richiedi_break_retest"].get())
        r.richiedi_vol_spike = bool(self.rule_vars["richiedi_vol_spike"].get())

    def _validate_add(self, add_size: float, add_price: float):
        self._sync_rules_from_ui()
        p = self.pos
        r = self.rules
        reasons = []

        if add_size <= 0:
            reasons.append("Size aggiunta non valida (<= 0).")
        if add_price <= 0:
            reasons.append("Prezzo aggiunta non valido (<= 0).")
        if p.prezzo_medio <= 0 or p.size_eth <= 0:
            reasons.append("Inserisci prezzo medio e size della posizione prima di proporre aggiunte.")
        if p.aggiunte_effettuate >= r.max_aggiunte:
            reasons.append(f"Superato max aggiunte ({r.max_aggiunte}).")
        if p.ultimo_add_ts > 0:
            mins = (time.time() - p.ultimo_add_ts) / 60.0
            if mins < r.cooldown_minuti:
                reasons.append(f"Cooldown non rispettato: {mins:.1f} min < {r.cooldown_minuti} min.")
        if p.ultimo_add_price > 0:
            dist = abs(pct(p.ultimo_add_price, add_price)) * 100.0
            if dist < r.distanza_min_pct:
                reasons.append(f"Distanza minima non rispettata: {dist:.2f}% < {r.distanza_min_pct:.2f}%.")
        if p.liq_price > 0:
            dist_liq = (add_price - p.liq_price) / add_price * 100.0
            if dist_liq < r.margine_min_pct_da_liq:
                reasons.append(f"Troppo vicino a liquidation: {dist_liq:.2f}% < {r.margine_min_pct_da_liq:.2f}%.")
        if p.stop_price > 0 and p.equity_usdt > 0:
            new_size = p.size_eth + add_size
            new_avg = (p.prezzo_medio * p.size_eth + add_price * add_size) / new_size
            risk_usdt = max(0.0, (new_avg - p.stop_price) * new_size)
            risk_pct = (risk_usdt / p.equity_usdt) * 100.0
            if risk_pct > r.rischio_max_pct_equity:
                reasons.append(f"Rischio su stop troppo alto dopo l'aggiunta: {risk_pct:.2f}% > {r.rischio_max_pct_equity:.2f}%.")
        if r.richiedi_sweep_reclaim and not self.confirm_sweep.get():
            reasons.append("Manca conferma sweep+reclaim.")
        if r.richiedi_break_retest and not self.confirm_retest.get():
            reasons.append("Manca conferma break+retest.")
        if r.richiedi_vol_spike and not self.confirm_vol.get():
            reasons.append("Manca conferma spike volume.")

        return (len(reasons) == 0), reasons

    def _spiegazione_rifiuto(self, reasons, add_size, add_price) -> str:
        p = self.pos
        r = self.rules
        tips = []
        for reason in reasons:
            if "max aggiunte" in reason:
                tips.append("• Hai già finito le aggiunte consentite. Ora è disciplina: non aggiungere.")
            elif "Cooldown" in reason:
                tips.append(f"• Troppo presto. Aspetta almeno {r.cooldown_minuti} minuti dall'ultima aggiunta.")
            elif "Distanza minima" in reason:
                tips.append(f"• Media troppo ravvicinata (tipico martingala). Aspetta distanza ≥ {r.distanza_min_pct:.2f}% o struttura.")
            elif "Troppo vicino a liquidation" in reason:
                tips.append(f"• Sei troppo vicino alla liquidation. Aggiungere size aumenta il rischio di wipe. Riduci rischio o aspetta rimbalzo.")
            elif "Rischio su stop" in reason:
                tips.append(f"• Con questa aggiunta superi il rischio massimo. Riduci size proposta o NON aggiungere.")
            elif "sweep+reclaim" in reason:
                tips.append("• Non hai confermato sweep+reclaim: aspetta wick sotto livello + chiusura sopra.")
            elif "break+retest" in reason:
                tips.append("• Non hai confermato break+retest: aspetta chiusura sopra + retest tenuto.")
            elif "spike volume" in reason:
                tips.append("• Non hai confermato spike volume: evita di mediare nel mezzo del range.")
            else:
                tips.append(f"• {reason}")

        return (
            f"Richiesta: +{add_size:.4f} ETH @ {add_price:,.2f}\n"
            "Esito: BLOCCATO\n\n"
            "Motivi tecnici:\n" + "\n".join(f"- {x}" for x in reasons) +
            "\n\nSpiegazione / cosa fare adesso:\n" + "\n".join(tips) +
            "\n\nRegola d'oro: se stai mediando per ansia, questo rifiuto ti sta salvando."
        )

    def propose_add(self):
        self.apply_position()
        add_size = safe_float(self.add_size_var.get(), 0.0)
        add_price = safe_float(self.add_price_var.get(), 0.0) or self.pos.prezzo_attuale
        ok, reasons = self._validate_add(add_size, add_price)
        action = f"Proposta add: +{add_size:.4f} ETH @ {add_price:,.2f}"

        if ok:
            p = self.pos
            new_size = p.size_eth + add_size
            new_avg = (p.prezzo_medio * p.size_eth + add_price * add_size) / new_size if new_size > 0 else p.prezzo_medio
            p.size_eth = new_size
            p.prezzo_medio = new_avg
            p.aggiunte_effettuate += 1
            p.ultimo_add_ts = time.time()
            p.ultimo_add_price = add_price
            self.vars["size_eth"].set(f"{p.size_eth:.6f}")
            self.vars["prezzo_medio"].set(f"{p.prezzo_medio:.2f}")
            self._log_event(action, "CONSENTITO", "Tutte le regole rispettate.")
            self.refresh_metrics()
            self.status_var.set("Aggiunta CONSENTITA e registrata.")
            messagebox.showinfo("OK", "Aggiunta CONSENTITA e registrata.")
        else:
            spiegazione = self._spiegazione_rifiuto(reasons, add_size, add_price)
            self._log_event(action, "BLOCCATO", " | ".join(reasons))
            self.status_var.set("Aggiunta BLOCCATA (vedi popup).")
            messagebox.showwarning("BLOCCATO (Anti-Martingala)", spiegazione)

    def _log_event(self, azione: str, esito: str, motivo: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = (ts, azione, esito, motivo)
        self.log_rows.append(row)
        self.log_tree.insert("", tk.END, values=row)

    def add_note(self):
        win = tk.Toplevel(self)
        win.title("Aggiungi nota")
        win.geometry("520x260")
        ttk.Label(win, text="Scrivi una nota:").pack(anchor="w", padx=10, pady=(10, 6))
        txt = tk.Text(win, height=8, wrap=tk.WORD)
        txt.pack(fill=tk.BOTH, expand=True, padx=10)
        def save():
            note = txt.get("1.0", tk.END).strip()
            if note:
                self._log_event("NOTA", "INFO", note)
                win.destroy()
        ttk.Button(win, text="Salva nota", command=save).pack(pady=10)

    def clear_log(self):
        if messagebox.askyesno("Conferma", "Vuoi cancellare il log della sessione?"):
            self.log_rows.clear()
            for item in self.log_tree.get_children():
                self.log_tree.delete(item)
            self.status_var.set("Log pulito.")

    def export_log(self):
        if not self.log_rows:
            messagebox.showinfo("Info", "Nessun log da esportare.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")], initialfile="guardiano_log.csv")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["ts", "azione", "esito", "motivo"])
            w.writerows(self.log_rows)
        self.status_var.set(f"Log esportato: {path}")

    def save_rules(self):
        self._sync_rules_from_ui()
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")], initialfile="regole_anti_martingala.json")
        if not path:
            return
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.rules), f, ensure_ascii=False, indent=2)
        self.status_var.set("Regole salvate.")

    def load_rules(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path:
            return
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.items():
            if hasattr(self.rules, k):
                setattr(self.rules, k, v)
        self.rule_vars["max_aggiunte"].set(int(self.rules.max_aggiunte))
        self.rule_vars["cooldown_minuti"].set(int(self.rules.cooldown_minuti))
        self.rule_vars["distanza_min_pct"].set(float(self.rules.distanza_min_pct))
        self.rule_vars["rischio_max_pct_equity"].set(float(self.rules.rischio_max_pct_equity))
        self.rule_vars["margine_min_pct_da_liq"].set(float(self.rules.margine_min_pct_da_liq))
        self.rule_vars["richiedi_sweep_reclaim"].set(bool(self.rules.richiedi_sweep_reclaim))
        self.rule_vars["richiedi_break_retest"].set(bool(self.rules.richiedi_break_retest))
        self.rule_vars["richiedi_vol_spike"].set(bool(self.rules.richiedi_vol_spike))
        self.status_var.set("Regole caricate.")

    def reset_session(self):
        if not messagebox.askyesno("Conferma", "Resetta la sessione?"):
            return
        self.pos.aggiunte_effettuate = 0
        self.pos.ultimo_add_ts = 0.0
        self.pos.ultimo_add_price = 0.0
        self.confirm_sweep.set(False)
        self.confirm_retest.set(False)
        self.confirm_vol.set(False)
        self.refresh_metrics()
        self.status_var.set("Sessione resettata.")

    def _fetch_price_binance(self, symbol: str):
        if requests is None:
            return None
        try:
            r = requests.get("https://api.binance.com/api/v3/ticker/price", params={"symbol": symbol.upper()}, timeout=3)
            r.raise_for_status()
            return float(r.json()["price"])
        except Exception:
            return None

    def _loop(self):
        try:
            if self.price_feed_enabled.get():
                now = time.time()
                if now - self.last_price_fetch >= max(2, int(self.auto_update_secs.get())):
                    self.last_price_fetch = now
                    sym = self.symbol_var.get().strip() or DEFAULT_SYMBOL
                    price = self._fetch_price_binance(sym)
                    if price is not None:
                        self.pos.prezzo_attuale = price
                        self.vars["prezzo_attuale"].set(f"{price:.2f}")
                        self.refresh_metrics()
                        self.status_var.set(f"Prezzo live {sym.upper()}: {price:.2f}")
        finally:
            self.after(500, self._loop)

def main():
    GuardianoApp().mainloop()

if __name__ == "__main__":
    main()
