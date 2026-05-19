"""
TBRGS graphical interface — origin/destination, model, timestamp, route display.
"""
import os
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

import yaml

from route_search import top_k_paths, format_route_summary

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    return {"defaults": {}}


class TBRGSApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TBRGS — Boroondara Route Guidance")
        self.geometry("720x520")
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        cfg = load_config()
        d = cfg.get("defaults", {})

        frm = ttk.Frame(self, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frm, text="Origin (SCATS ID):").grid(row=0, column=0, sticky=tk.W, pady=4)
        self.origin_var = tk.StringVar(value=str(d.get("origin", 2000)))
        ttk.Entry(frm, textvariable=self.origin_var, width=12).grid(row=0, column=1, sticky=tk.W)

        ttk.Label(frm, text="Destination (SCATS ID):").grid(row=1, column=0, sticky=tk.W, pady=4)
        self.dest_var = tk.StringVar(value=str(d.get("destination", 3002)))
        ttk.Entry(frm, textvariable=self.dest_var, width=12).grid(row=1, column=1, sticky=tk.W)

        ttk.Label(frm, text="Date/time (YYYY-MM-DD HH:MM):").grid(row=2, column=0, sticky=tk.W, pady=4)
        self.time_var = tk.StringVar(value=d.get("timestamp", "2006-10-27 08:00"))
        ttk.Entry(frm, textvariable=self.time_var, width=22).grid(row=2, column=1, sticky=tk.W)

        ttk.Label(frm, text="ML model:").grid(row=3, column=0, sticky=tk.W, pady=4)
        self.model_var = tk.StringVar(value=d.get("model", "lstm"))
        ttk.Combobox(
            frm, textvariable=self.model_var, values=["lstm", "gru", "cnn"], width=10, state="readonly"
        ).grid(row=3, column=1, sticky=tk.W)

        ttk.Label(frm, text="Top-k routes:").grid(row=4, column=0, sticky=tk.W, pady=4)
        self.k_var = tk.StringVar(value=str(d.get("top_k", 5)))
        ttk.Spinbox(frm, from_=1, to=5, textvariable=self.k_var, width=6).grid(row=4, column=1, sticky=tk.W)

        ttk.Button(frm, text="Find routes", command=self.on_find).grid(row=5, column=0, columnspan=2, pady=12)

        self.output = scrolledtext.ScrolledText(frm, height=18, width=80, font=("Menlo", 11))
        self.output.grid(row=6, column=0, columnspan=3, sticky="nsew")
        frm.rowconfigure(6, weight=1)
        frm.columnconfigure(2, weight=1)

    def on_find(self):
        try:
            o = int(self.origin_var.get())
            d = int(self.dest_var.get())
            ts = self.time_var.get().strip()
            model = self.model_var.get()
            k = int(self.k_var.get())
        except ValueError:
            messagebox.showerror("Input error", "Check origin, destination, and top-k values.")
            return

        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, "Searching...\n")
        self.update()

        try:
            routes = top_k_paths(o, d, ts, k=k, model=model)
            text = format_route_summary(routes, o, d)
            text += f"\n\nModel: {model.upper()}  |  Time: {ts}"
            self.output.delete("1.0", tk.END)
            self.output.insert(tk.END, text)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.output.delete("1.0", tk.END)
            self.output.insert(tk.END, str(e))


if __name__ == "__main__":
    TBRGSApp().mainloop()
