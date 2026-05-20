"""
gui.py – TBRGS Graphical User Interface

Run from the 2B/src/ directory:
    python gui.py

Requirements: tkinter (stdlib), matplotlib, networkx, pandas, numpy, torch, sklearn
"""

import os
import sys
import json
import threading

# ── working-directory fix so every relative path resolves ──────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import numpy as np
import pandas as pd
import networkx as nx

from graph import build_graph
from router import find_top_k_routes
from predict import predict_flow

# ── paths ───────────────────────────────────────────────────────────
CONFIG_PATH = os.path.join(os.path.dirname(_SCRIPT_DIR), "config.json")

MODEL_PATHS = {
    "lstm": "models/lstm_best.pth",
    "gru":  "models/gru_best.pth",
    "cnn":  "models/cnn_best.pth",
}
MODEL_LOSS_PATHS = {
    "lstm": "models/lstm_best_losses.csv",
    "gru":  "models/gru_best_losses.csv",
    "cnn":  "models/cnn_best_losses.csv",
}
AVAILABLE_MODELS = list(MODEL_PATHS.keys())

# ── style constants ─────────────────────────────────────────────────
ROUTE_COLORS  = ["#e74c3c", "#2980b9", "#27ae60", "#8e44ad", "#e67e22"]
MODEL_COLORS  = {"lstm": "#2980b9", "gru": "#e74c3c", "cnn": "#27ae60"}
BG_LIGHT      = "#f4f6f9"
BG_MAP        = "#dde8f0"

DEFAULT_CONFIG = {
    "speed_limit_kmh":      60,
    "intersection_delay_s": 30,
    "max_routes":           5,
    "default_model":        "lstm",
    "max_edge_distance_km": 2.0,
    "default_timestamp":    "2006-10-27 08:00",
    "default_origin":       2000,
    "default_destination":  3002,
}


# ════════════════════════════════════════════════════════════════════
class TBRGSApp:
    """Main application window."""

    # ── init ────────────────────────────────────────────────────────
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("TBRGS – Traffic-Based Route Guidance System")
        self.root.geometry("1440x880")
        self.root.minsize(1100, 720)
        self.root.configure(bg=BG_LIGHT)

        self.config       = self._load_config()
        self.sites_df     = None
        self.graph        = None
        self.current_routes: list[dict] = []
        self.selected_route = tk.IntVar(value=0)

        self._setup_styles()
        self._build_ui()

        # Load heavy data in background after window appears
        self.root.after(100, lambda: threading.Thread(
            target=self._load_data, daemon=True).start())

    # ── config ──────────────────────────────────────────────────────
    def _load_config(self) -> dict:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH) as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        return dict(DEFAULT_CONFIG)

    def _save_config(self):
        updated = {
            "speed_limit_kmh":      self.speed_var.get(),
            "intersection_delay_s": self.delay_var.get(),
            "max_routes":           self.routes_var.get(),
            "default_model":        self.model_var.get(),
            "max_edge_distance_km": self.config["max_edge_distance_km"],
            "default_timestamp":    self.timestamp_var.get().strip(),
            "default_origin":       self._parse_id(self.origin_var.get()) or self.config["default_origin"],
            "default_destination":  self._parse_id(self.dest_var.get())   or self.config["default_destination"],
        }
        with open(CONFIG_PATH, "w") as f:
            json.dump(updated, f, indent=2)
        self.config = updated
        messagebox.showinfo("Saved", "Settings saved as defaults.")

    # ── styles ──────────────────────────────────────────────────────
    def _setup_styles(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure(".",                      background=BG_LIGHT, font=("Helvetica", 9))
        s.configure("TLabelframe",            background=BG_LIGHT)
        s.configure("TLabelframe.Label",      font=("Helvetica", 10, "bold"), background=BG_LIGHT)
        s.configure("TFrame",                 background=BG_LIGHT)
        s.configure("TLabel",                 background=BG_LIGHT)
        s.configure("TRadiobutton",           background=BG_LIGHT)
        s.configure("TCheckbutton",           background=BG_LIGHT)
        s.configure("Big.TButton",            font=("Helvetica", 10, "bold"), padding=6)
        s.configure("TNotebook",              background=BG_LIGHT)
        s.configure("TNotebook.Tab",          font=("Helvetica", 9, "bold"), padding=(10, 4))
        s.map("TNotebook.Tab", background=[("selected", "#ffffff")])

    # ── UI construction ─────────────────────────────────────────────
    def _build_ui(self):
        self.root.columnconfigure(0, weight=0, minsize=355)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)

        # Left sidebar
        sidebar = ttk.Frame(self.root, padding=(10, 10, 6, 4))
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.columnconfigure(0, weight=1)
        self._build_sidebar(sidebar)

        # Right main area (tabs)
        main = ttk.Frame(self.root, padding=(4, 10, 10, 4))
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)
        self._build_main_area(main)

        # Status bar
        bar = tk.Frame(self.root, bg="#2c3e50", pady=3)
        bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.status_var = tk.StringVar(value="Initialising…")
        tk.Label(bar, textvariable=self.status_var, bg="#2c3e50", fg="#ecf0f1",
                 font=("Helvetica", 9), anchor="w", padx=10).pack(fill="x")

    # ── sidebar ─────────────────────────────────────────────────────
    def _build_sidebar(self, parent):
        r = 0

        # Header
        tk.Label(parent, text="TBRGS", font=("Helvetica", 20, "bold"),
                 bg=BG_LIGHT, fg="#2c3e50").grid(row=r, column=0, sticky="w"); r += 1
        tk.Label(parent, text="Traffic-Based Route Guidance System",
                 font=("Helvetica", 8), bg=BG_LIGHT, fg="#7f8c8d").grid(
                     row=r, column=0, sticky="w", pady=(0, 12)); r += 1

        # ── Route Query ─────────────────────────────────────────────
        qf = ttk.LabelFrame(parent, text="Route Query", padding=10)
        qf.grid(row=r, column=0, sticky="ew", pady=(0, 8)); r += 1
        qf.columnconfigure(0, weight=1)

        ttk.Label(qf, text="Origin (SCATS ID):").grid(row=0, column=0, sticky="w")
        self.origin_var = tk.StringVar(value=str(self.config["default_origin"]))
        self.origin_combo = ttk.Combobox(qf, textvariable=self.origin_var)
        self.origin_combo.grid(row=1, column=0, sticky="ew", pady=(2, 8))
        self.origin_combo.bind("<<ComboboxSelected>>", lambda _: self._draw_network())

        ttk.Label(qf, text="Destination (SCATS ID):").grid(row=2, column=0, sticky="w")
        self.dest_var = tk.StringVar(value=str(self.config["default_destination"]))
        self.dest_combo = ttk.Combobox(qf, textvariable=self.dest_var)
        self.dest_combo.grid(row=3, column=0, sticky="ew", pady=(2, 8))
        self.dest_combo.bind("<<ComboboxSelected>>", lambda _: self._draw_network())

        ttk.Label(qf, text="Date & Time  (YYYY-MM-DD HH:MM):").grid(row=4, column=0, sticky="w")
        self.timestamp_var = tk.StringVar(value=self.config["default_timestamp"])
        ttk.Entry(qf, textvariable=self.timestamp_var).grid(row=5, column=0, sticky="ew", pady=(2, 8))

        ttk.Label(qf, text="Prediction Model:").grid(row=6, column=0, sticky="w")
        self.model_var = tk.StringVar(value=self.config["default_model"])
        mf = ttk.Frame(qf)
        mf.grid(row=7, column=0, sticky="ew", pady=(2, 10))
        for m in AVAILABLE_MODELS:
            exists = os.path.exists(MODEL_PATHS[m])
            ttk.Radiobutton(mf, text=m.upper(), variable=self.model_var,
                            value=m, state="normal" if exists else "disabled"
                            ).pack(side="left", padx=(0, 6))

        self.find_btn = ttk.Button(qf, text="Find Routes", style="Big.TButton",
                                   command=self._find_routes_async)
        self.find_btn.grid(row=8, column=0, sticky="ew")
        self.progress = ttk.Progressbar(qf, mode="indeterminate")
        self.progress.grid(row=9, column=0, sticky="ew", pady=(4, 0))

        # ── Parameters ──────────────────────────────────────────────
        pf = ttk.LabelFrame(parent, text="Parameters", padding=10)
        pf.grid(row=r, column=0, sticky="ew", pady=(0, 8)); r += 1
        pf.columnconfigure(1, weight=1)

        params = [
            ("Speed limit (km/h):",      "speed_var",  tk.IntVar,  20,  130, "speed_limit_kmh"),
            ("Intersection delay (s):",  "delay_var",  tk.IntVar,   0,  300, "intersection_delay_s"),
            ("Max routes (1–5):",        "routes_var", tk.IntVar,   1,    5, "max_routes"),
        ]
        for idx, (label, attr, vtype, lo, hi, cfg_key) in enumerate(params):
            ttk.Label(pf, text=label).grid(row=idx, column=0, sticky="w", padx=(0, 8), pady=3)
            var = vtype(value=self.config[cfg_key])
            setattr(self, attr, var)
            ttk.Spinbox(pf, from_=lo, to=hi, textvariable=var, width=8).grid(
                row=idx, column=1, sticky="w", pady=3)

        ttk.Button(pf, text="Save as Default", command=self._save_config).grid(
            row=len(params), column=0, columnspan=2, sticky="ew", pady=(8, 0))

        # ── Results ─────────────────────────────────────────────────
        rf = ttk.LabelFrame(parent, text="Results", padding=10)
        rf.grid(row=r, column=0, sticky="nsew", pady=(0, 8)); r += 1
        rf.columnconfigure(0, weight=1)
        rf.rowconfigure(1, weight=1)
        parent.rowconfigure(r - 1, weight=1)

        # Route radio-selector strip
        self.route_strip = ttk.Frame(rf)
        self.route_strip.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        # Scrollable detail text
        txt_wrap = ttk.Frame(rf)
        txt_wrap.grid(row=1, column=0, sticky="nsew")
        txt_wrap.columnconfigure(0, weight=1)
        txt_wrap.rowconfigure(0, weight=1)

        vsb = ttk.Scrollbar(txt_wrap)
        vsb.grid(row=0, column=1, sticky="ns")
        self.results_text = tk.Text(
            txt_wrap, height=10, font=("Courier", 9),
            wrap="word", state="disabled",
            yscrollcommand=vsb.set, padx=6, pady=4,
            bg="#ffffff", relief="flat", borderwidth=1,
        )
        self.results_text.grid(row=0, column=0, sticky="nsew")
        vsb.config(command=self.results_text.yview)

        # Text tags
        self.results_text.tag_config("title",  font=("Helvetica", 10, "bold"))
        self.results_text.tag_config("info",   font=("Courier", 8),  foreground="#555555")
        self.results_text.tag_config("path",   font=("Courier", 8),  foreground="#2c3e50")
        for i, col in enumerate(ROUTE_COLORS, 1):
            self.results_text.tag_config(f"r{i}", foreground=col, font=("Helvetica", 9, "bold"))

    # ── main area (notebook) ────────────────────────────────────────
    def _build_main_area(self, parent):
        self.notebook = ttk.Notebook(parent)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        # ── Tab 1: Network Map ──────────────────────────────────────
        t1 = ttk.Frame(self.notebook)
        self.notebook.add(t1, text="  Network Map  ")
        t1.columnconfigure(0, weight=1)
        t1.rowconfigure(0, weight=1)

        self.fig_map, self.ax_map = plt.subplots(figsize=(9, 6.2))
        self.fig_map.set_facecolor(BG_MAP)
        self.canvas_map = FigureCanvasTkAgg(self.fig_map, t1)
        self.canvas_map.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        tb1 = ttk.Frame(t1)
        tb1.grid(row=1, column=0, sticky="ew")
        NavigationToolbar2Tk(self.canvas_map, tb1)

        # ── Tab 2: Traffic Prediction ───────────────────────────────
        t2 = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(t2, text="  Traffic Prediction  ")
        t2.columnconfigure(2, weight=1)
        t2.rowconfigure(1, weight=1)

        ctrl = ttk.Frame(t2)
        ctrl.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 8))

        ttk.Label(ctrl, text="SCATS Site:").pack(side="left", padx=(0, 4))
        self.pred_site_var = tk.StringVar()
        self.pred_site_combo = ttk.Combobox(ctrl, textvariable=self.pred_site_var, width=28)
        self.pred_site_combo.pack(side="left", padx=(0, 8))

        ttk.Label(ctrl, text="Day (YYYY-MM-DD):").pack(side="left", padx=(0, 4))
        self.pred_date_var = tk.StringVar(value="2006-10-27")
        ttk.Entry(ctrl, textvariable=self.pred_date_var, width=12).pack(side="left", padx=(0, 8))

        ttk.Button(ctrl, text="Plot", command=self._plot_prediction).pack(side="left")

        self.fig_pred, self.ax_pred = plt.subplots(figsize=(9, 5.2))
        self.fig_pred.set_facecolor(BG_LIGHT)
        self.canvas_pred = FigureCanvasTkAgg(self.fig_pred, t2)
        self.canvas_pred.get_tk_widget().grid(row=1, column=0, columnspan=4, sticky="nsew")
        tb2 = ttk.Frame(t2)
        tb2.grid(row=2, column=0, columnspan=4, sticky="ew")
        NavigationToolbar2Tk(self.canvas_pred, tb2)

        # ── Tab 3: Model Comparison ─────────────────────────────────
        t3 = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(t3, text="  Model Comparison  ")
        t3.columnconfigure(0, weight=1)
        t3.rowconfigure(0, weight=1)

        self.fig_comp, self.axes_comp = plt.subplots(1, 2, figsize=(9, 5))
        self.fig_comp.set_facecolor(BG_LIGHT)
        self.canvas_comp = FigureCanvasTkAgg(self.fig_comp, t3)
        self.canvas_comp.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        tb3 = ttk.Frame(t3)
        tb3.grid(row=1, column=0, sticky="ew")
        NavigationToolbar2Tk(self.canvas_comp, tb3)

        # Draw static model comparison immediately (uses saved CSV files)
        self._plot_model_comparison()

    # ── data loading ────────────────────────────────────────────────
    def _load_data(self):
        try:
            df = pd.read_csv("data/site_info.csv")
            df["scats_id"] = df["scats_id"].astype(int)

            labels = [
                f"{int(row['scats_id'])}  –  {row['location']}"
                for _, row in df.iterrows()
            ]
            G, _ = build_graph()

            self.root.after(0, self._on_data_loaded, df, labels, G)
        except Exception as e:
            self.root.after(0, self._set_status, f"Load error: {e}")

    def _on_data_loaded(self, df, labels, G):
        self.sites_df = df
        self.graph    = G

        for cb in (self.origin_combo, self.dest_combo,
                   self.pred_site_combo):
            cb.config(values=labels)

        # Restore default O/D as rich labels
        def label_for(sid):
            row = df[df["scats_id"] == int(sid)]
            if row.empty:
                return str(sid)
            r = row.iloc[0]
            return f"{int(r['scats_id'])}  –  {r['location']}"

        self.origin_var.set(label_for(self.config["default_origin"]))
        self.dest_var.set(label_for(self.config["default_destination"]))

        self._draw_network()
        self._set_status(
            f"Ready — {len(df)} SCATS sites, "
            f"{G.number_of_edges()} road links loaded."
        )

    # ── helpers ─────────────────────────────────────────────────────
    def _parse_id(self, val: str) -> int | None:
        """Extract numeric SCATS ID from '2000  –  WARRIGAL_RD…' or '2000'."""
        try:
            return int(str(val).split("–")[0].strip())
        except Exception:
            return None

    def _set_status(self, msg: str):
        self.status_var.set(msg)

    # ── network map ─────────────────────────────────────────────────
    def _draw_network(self, highlight_paths: list | None = None):
        if self.sites_df is None or self.graph is None:
            return

        ax = self.ax_map
        ax.clear()
        ax.set_facecolor(BG_MAP)

        pos = {
            int(r["scats_id"]): (r["lon"], r["lat"])
            for _, r in self.sites_df.iterrows()
        }

        # Base graph
        nx.draw_networkx_edges(
            self.graph, pos, ax=ax,
            alpha=0.30, edge_color="#7f8c8d", width=1.2,
        )
        nx.draw_networkx_nodes(
            self.graph, pos, ax=ax,
            node_size=28, node_color="#2980b9", alpha=0.70,
        )
        nx.draw_networkx_labels(
            self.graph, pos, ax=ax,
            font_size=5, font_color="#1c1c1c",
        )

        patches = []

        # Highlighted routes
        if highlight_paths:
            for i, path in enumerate(highlight_paths):
                col = ROUTE_COLORS[i % len(ROUTE_COLORS)]
                edges = [
                    (path[j], path[j + 1])
                    for j in range(len(path) - 1)
                    if self.graph.has_edge(path[j], path[j + 1])
                    or self.graph.has_edge(path[j + 1], path[j])
                ]
                nx.draw_networkx_edges(
                    self.graph, pos, edgelist=edges, ax=ax,
                    edge_color=col, width=4.5, alpha=0.85,
                )
                patches.append(mpatches.Patch(color=col, label=f"Route {i + 1}"))

        # Origin / Destination markers
        origin_id = self._parse_id(self.origin_var.get())
        dest_id   = self._parse_id(self.dest_var.get())

        for node_id, col, lbl in [
            (origin_id, "#27ae60", "Origin"),
            (dest_id,   "#e74c3c", "Dest"),
        ]:
            if node_id and node_id in pos:
                nx.draw_networkx_nodes(
                    self.graph, pos, nodelist=[node_id], ax=ax,
                    node_color=col, node_size=200, alpha=1.0,
                )
                patches.append(mpatches.Patch(color=col, label=f"{lbl} ({node_id})"))

        if patches:
            ax.legend(handles=patches, loc="upper left",
                      fontsize=8, framealpha=0.92, edgecolor="#ccc")

        ax.set_title("Boroondara Road Network", fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Longitude",  fontsize=8)
        ax.set_ylabel("Latitude",   fontsize=8)
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True, labelsize=7)
        self.fig_map.tight_layout()
        self.canvas_map.draw()

    # ── route finding ───────────────────────────────────────────────
    def _find_routes_async(self):
        self.find_btn.config(state="disabled")
        self.progress.start()
        self._set_status("Computing routes…")
        threading.Thread(target=self._find_routes, daemon=True).start()

    def _find_routes(self):
        try:
            origin    = self._parse_id(self.origin_var.get())
            dest      = self._parse_id(self.dest_var.get())
            timestamp = self.timestamp_var.get().strip()
            model     = self.model_var.get()
            k         = self.routes_var.get()
            spd       = float(self.speed_var.get())
            delay     = float(self.delay_var.get())

            if not origin or not dest:
                raise ValueError("Please select a valid origin and destination.")
            if origin == dest:
                raise ValueError("Origin and destination must be different sites.")

            routes = find_top_k_routes(
                origin, dest, timestamp,
                model=model, k=k,
                speed_limit=spd, intersection_delay=delay,
            )
            self.root.after(0, self._display_results, routes, origin, dest)
        except Exception as exc:
            self.root.after(0, messagebox.showerror, "Route Error", str(exc))
            self.root.after(0, self._set_status, "Route search failed.")
        finally:
            self.root.after(0, self._search_done)

    def _search_done(self):
        self.progress.stop()
        self.find_btn.config(state="normal")

    def _display_results(self, routes: list[dict], origin: int, dest: int):
        self.current_routes = routes

        # Clear route strip
        for w in self.route_strip.winfo_children():
            w.destroy()

        self.results_text.config(state="normal")
        self.results_text.delete("1.0", "end")

        if not routes:
            self.results_text.insert("end", f"No route found from {origin} to {dest}.\n\n")
            self.results_text.insert(
                "end",
                "Possible reasons:\n"
                "  • Sites not connected via shared road segments\n"
                "  • Max edge distance too small (edit config.json → max_edge_distance_km)\n",
                "info",
            )
            self._draw_network()
            self._set_status(f"No routes found: {origin} → {dest}.")
        else:
            self.results_text.insert("end", f"Top {len(routes)} route(s): {origin} → {dest}\n\n", "title")

            for i, route in enumerate(routes, 1):
                mins  = route["travel_time"] / 60
                col_tag = f"r{i}"

                # Radio button in strip
                ttk.Radiobutton(
                    self.route_strip,
                    text=f"Route {i} ({mins:.1f} min)",
                    variable=self.selected_route,
                    value=i - 1,
                    command=self._highlight_selected,
                ).pack(side="left", padx=4)

                # Detail text
                self.results_text.insert("end", f"  Route {i}", col_tag)
                self.results_text.insert(
                    "end",
                    f"  ─  {mins:.1f} min  |  "
                    f"{route['num_intersections']} intersection(s)\n",
                    "info",
                )
                path_str = " → ".join(str(s) for s in route["path"])
                self.results_text.insert("end", f"  {path_str}\n\n", "path")

            # Select first route
            self.selected_route.set(0)
            self._draw_network(highlight_paths=[r["path"] for r in routes])
            self.notebook.select(0)
            self._set_status(
                f"Found {len(routes)} route(s) from {origin} to {dest}. "
                f"Best: {routes[0]['travel_time'] / 60:.1f} min."
            )

        self.results_text.config(state="disabled")

    def _highlight_selected(self):
        idx = self.selected_route.get()
        if idx < len(self.current_routes):
            self._draw_network(highlight_paths=[self.current_routes[idx]["path"]])

    # ── traffic prediction chart ────────────────────────────────────
    def _plot_prediction(self):
        site_id = self._parse_id(self.pred_site_var.get())
        day_str = self.pred_date_var.get().strip()
        if not site_id:
            messagebox.showwarning("No site", "Select a SCATS site first.")
            return
        self._set_status(f"Computing predictions for site {site_id} on {day_str}…")
        threading.Thread(
            target=self._compute_prediction_plot,
            args=(site_id, day_str),
            daemon=True,
        ).start()

    def _compute_prediction_plot(self, site_id: int, day_str: str):
        try:
            # Load all data for this site
            dfs = []
            for path in ("data/train.csv", "data/val.csv", "data/test.csv"):
                df = pd.read_csv(path)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                dfs.append(df)
            all_data = pd.concat(dfs)
            site_data = (
                all_data[all_data["scats_id"] == site_id]
                .sort_values("timestamp")
            )

            # Filter to the selected day
            target_date = pd.Timestamp(day_str).normalize()
            day_data = site_data[
                site_data["timestamp"].dt.normalize() == target_date
            ]

            if day_data.empty:
                self.root.after(
                    0, messagebox.showwarning,
                    "No data",
                    f"No data for site {site_id} on {day_str}.\n"
                    "Available range: 2006-10-01 to 2006-10-31.",
                )
                return

            timestamps = day_data["timestamp"].tolist()
            actual     = day_data["flow_15min"].values

            predictions: dict[str, list] = {}
            for m in AVAILABLE_MODELS:
                if not os.path.exists(MODEL_PATHS[m]):
                    continue
                preds = []
                for ts in timestamps:
                    try:
                        preds.append(predict_flow(site_id, ts, model=m))
                    except Exception:
                        preds.append(float("nan"))
                predictions[m] = preds

            self.root.after(
                0, self._draw_prediction_chart,
                timestamps, actual, predictions, site_id, day_str,
            )
        except Exception as exc:
            self.root.after(0, messagebox.showerror, "Prediction Error", str(exc))
        finally:
            self.root.after(0, self._set_status, "Ready")

    def _draw_prediction_chart(self, timestamps, actual, predictions, site_id, day_str):
        ax = self.ax_pred
        ax.clear()
        ax.set_facecolor(BG_LIGHT)

        ax.plot(timestamps, actual, color="#2c3e50", linewidth=2,
                label="Actual", alpha=0.9, zorder=5)

        for m, preds in predictions.items():
            ax.plot(timestamps, preds,
                    color=MODEL_COLORS.get(m, "gray"),
                    linewidth=1.5, linestyle="--",
                    label=m.upper(), alpha=0.85)

        # Site location label
        if self.sites_df is not None:
            row = self.sites_df[self.sites_df["scats_id"] == site_id]
            loc = row.iloc[0]["location"] if not row.empty else ""
        else:
            loc = ""

        ax.set_title(
            f"Traffic Flow Prediction – Site {site_id}: {loc}\n{day_str}",
            fontsize=11, fontweight="bold",
        )
        ax.set_xlabel("Time of Day", fontsize=9)
        ax.set_ylabel("Flow (vehicles / 15 min)", fontsize=9)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        self.fig_pred.tight_layout()
        self.canvas_pred.draw()
        self.notebook.select(1)

    # ── model comparison chart ──────────────────────────────────────
    def _plot_model_comparison(self):
        ax_tr, ax_va = self.axes_comp
        ax_tr.set_facecolor(BG_LIGHT)
        ax_va.set_facecolor(BG_LIGHT)

        loaded = False
        for m, path in MODEL_LOSS_PATHS.items():
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            col = MODEL_COLORS.get(m, "gray")
            ax_tr.plot(df["epoch"], df["train_loss"], color=col,
                       linewidth=2, label=m.upper())
            ax_va.plot(df["epoch"], df["val_loss"],   color=col,
                       linewidth=2, linestyle="--", label=m.upper())
            loaded = True

        if loaded:
            for ax, title in [(ax_tr, "Training Loss (MSE)"),
                              (ax_va, "Validation Loss (MSE)")]:
                ax.set_title(title, fontweight="bold", fontsize=10)
                ax.set_xlabel("Epoch", fontsize=9)
                ax.set_ylabel("MSE Loss", fontsize=9)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
        else:
            ax_tr.text(0.5, 0.5, "No loss data found.\nTrain models first.",
                       ha="center", va="center", transform=ax_tr.transAxes,
                       fontsize=11, color="#888")
            ax_va.axis("off")

        self.fig_comp.suptitle("ML Model Comparison – Training Curves",
                               fontsize=12, fontweight="bold")
        self.fig_comp.tight_layout()
        self.canvas_comp.draw()


# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app  = TBRGSApp(root)
    root.mainloop()
