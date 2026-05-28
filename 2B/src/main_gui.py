"""
main_gui.py
===========
TBRGS — Traffic-Based Route Guidance System
Desktop GUI built with PyQt5 and Folium / OpenStreetMap.

Layout
------
  Left panel  : SCATS site selection, model picker, Find button, results
  Right panel : Interactive Folium map embedded via QWebEngineView

Usage
-----
    python main_gui.py
"""
import sys
import os
import folium

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QTextEdit, QFrame,
    QMessageBox, QSizePolicy, QGroupBox, QScrollArea,
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont

from routing import get_top_k_routes, build_traffic_graph

# ---------------------------------------------------------------------------
# Load SCATS site metadata
# ---------------------------------------------------------------------------
_SITE_INFO_PATH = os.path.join(SCRIPT_DIR, "data", "site_info.csv")

# Coordinate correction: AGD84 → WGS84 (applied in search_graph.py too,
# but we duplicate it here so the map markers match the routing graph).
LAT_OFFSET = -0.0011
LON_OFFSET =  0.0010

def _load_real_scats() -> dict:
    """
    Return {scats_id: (lat, lon, location_label)} from site_info.csv.
    Falls back to an empty dict if the file is missing.
    """
    if not os.path.exists(_SITE_INFO_PATH):
        print(f"[GUI] Warning: {_SITE_INFO_PATH} not found.")
        return {}
    df = pd.read_csv(_SITE_INFO_PATH)
    return {
        int(row["scats_id"]): (
            float(row["lat"])      + LAT_OFFSET,
            float(row["lon"])      + LON_OFFSET,
            str(row.get("location", f"Site {int(row['scats_id'])}")),
        )
        for _, row in df.iterrows()
    }

REAL_SCATS = _load_real_scats()

# ---------------------------------------------------------------------------
# Route colour palette
# ---------------------------------------------------------------------------
ROUTE_COLORS = ["red", "blue", "green", "purple", "orange"]
ROUTE_HEX    = {
    "red":    "#ff1900",
    "blue":   "#56b3f1",
    "green":  "#00c351",
    "purple": "#984fb7",
    "orange": "#ff973c",
}

MAP_OUTPUT = os.path.join(SCRIPT_DIR, "map.html")


# ===========================================================================
# Background worker thread
# ===========================================================================

class RoutingWorker(QThread):
    """
    Runs get_top_k_routes() in a background thread so the GUI stays
    responsive during flow prediction and path search.
    """
    finished = pyqtSignal(list)   # emits List[(path, travel_time_min)]
    error    = pyqtSignal(str)

    def __init__(self, origin: int, destination: int, model_name: str) -> None:
        super().__init__()
        self.origin      = origin
        self.destination = destination
        self.model_name  = model_name

    def run(self) -> None:
        try:
            routes = get_top_k_routes(
                self.origin, self.destination, self.model_name, k=5
            )
            if not routes:
                self.error.emit(
                    "No path found between the selected SCATS sites.\n"
                    "Try a different origin/destination pair."
                )
            else:
                self.finished.emit(routes)
        except ValueError as e:
            self.error.emit(str(e))
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}")


# ===========================================================================
# Folium map generation
# ===========================================================================

def generate_map(routes: list, origin: int, destination: int) -> str:
    """
    Render up to 5 routes on an OpenStreetMap base using Folium.

    Each route is drawn as a colour-coded polyline with a tooltip and
    popup showing path + estimated travel time.  The map is saved to
    MAP_OUTPUT and its absolute path is returned.
    """
    involved  = {n for path, _ in routes for n in path}
    coords    = [REAL_SCATS[n][:2] for n in involved if n in REAL_SCATS]
    if coords:
        center = (
            sum(c[0] for c in coords) / len(coords),
            sum(c[1] for c in coords) / len(coords),
        )
    else:
        center = (-37.858, 145.070)   # Boroondara fallback

    fmap = folium.Map(location=list(center), zoom_start=14, tiles="OpenStreetMap")

    # ── Route polylines ───────────────────────────────────────────────────
    for rank, (path, tt) in enumerate(routes):
        color  = ROUTE_COLORS[rank % len(ROUTE_COLORS)]
        points = [
            (REAL_SCATS[n][0], REAL_SCATS[n][1])
            for n in path if n in REAL_SCATS
        ]
        if len(points) < 2:
            continue

        label = " → ".join(map(str, path))
        folium.PolyLine(
            locations    = points,
            color        = color,
            weight       = 6 if rank == 0 else 4,
            opacity      = 0.95 if rank == 0 else 0.75,
            dash_array   = None if rank == 0 else ("10 5" if rank % 2 else None),
            tooltip      = f"Route {rank+1} [{color}]  |  {tt:.1f} min",
            popup        = folium.Popup(
                f"<div style='font-family:sans-serif'>"
                f"<b>Route {rank+1}</b> "
                f"<span style='color:{ROUTE_HEX[color]}'>({color})</span><br>"
                f"<b>Path:</b> {label}<br>"
                f"<b>Est. time:</b> {tt:.2f} min"
                f"</div>",
                max_width=320,
            ),
        ).add_to(fmap)

        # Direction arrows at midpoints
        for i in range(len(points) - 1):
            mid = ((points[i][0] + points[i+1][0]) / 2,
                   (points[i][1] + points[i+1][1]) / 2)
            folium.Marker(
                location=list(mid),
                icon=folium.DivIcon(
                    html=(
                        f'<div style="color:{ROUTE_HEX[color]};'
                        f'font-size:14px;font-weight:bold;">▶</div>'
                    ),
                    icon_size=(16, 16), icon_anchor=(8, 8),
                ),
            ).add_to(fmap)

    # ── SCATS site markers ────────────────────────────────────────────────
    for nid, (lat, lon, lbl) in REAL_SCATS.items():
        # Determine marker colour and icon based on origin/destination
        if nid == origin:
            bg_color = "#2ecc71"   # green
            icon_char = "🚦"       # or "🏁" etc.
        elif nid == destination:
            bg_color = "#e74c3c"   # red
            icon_char = "🏁"
        else:
            bg_color = "#3498db"   # blue
            icon_char = "📍"
        
        # Create a DivIcon with the SCATS number displayed prominently
        html = f'''
        <div style="
            background-color: {bg_color};
            color: white;
            font-weight: bold;
            font-size: 12px;
            font-family: Arial, sans-serif;
            text-align: center;
            line-height: 22px;
            width: 28px;
            height: 28px;
            border-radius: 50%;
            border: 2px solid white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        ">
            {nid}
        </div>
        '''
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(
                f"<b>SCATS {nid}</b><br>{lbl}",
                max_width=220
            ),
            tooltip=f"SCATS {nid} — {lbl}",
            icon=folium.DivIcon(html=html, icon_size=(28, 28), icon_anchor=(14, 14))
        ).add_to(fmap)

    # ── Legend ────────────────────────────────────────────────────────────
    rows = "".join(
        f'<tr><td><span style="color:{ROUTE_HEX[ROUTE_COLORS[i]]};font-size:18px;">'
        f'&#9644;</span></td>'
        f'<td style="padding-left:6px">Route {i+1} — {routes[i][1]:.1f} min</td></tr>'
        for i in range(len(routes))
    )
    fmap.get_root().html.add_child(folium.Element(f"""
    <div style="position:fixed;bottom:36px;left:36px;z-index:9999;
                background:rgba(255,255,255,0.95);padding:12px 16px;
                border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.2);
                font-family:'Segoe UI',Arial,sans-serif;font-size:13px;">
      <b style="font-size:14px">📍 Route Legend</b>
      <table style="margin-top:6px;border-collapse:collapse">{rows}</table>
      <div style="margin-top:8px;font-size:12px;color:#555">
        <span style="color:green">●</span> Origin (SCATS {origin}) &nbsp;
        <span style="color:red">●</span> Destination (SCATS {destination})
      </div>
    </div>
    """))

    fmap.save(MAP_OUTPUT)
    return os.path.realpath(MAP_OUTPUT)


# ===========================================================================
# Main Window
# ===========================================================================

class TBRGSWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("TBRGS — Traffic-Based Route Guidance System")
        self.setMinimumSize(1380, 820)
        self._worker: RoutingWorker | None = None
        self._build_ui()
        self._show_default_map()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_left_panel())
        root.addWidget(self._build_map_panel(), stretch=1)
        self.setStyleSheet(_STYLESHEET)

    def _build_left_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("leftPanel")
        panel.setFixedWidth(370)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(22, 26, 22, 18)
        layout.setSpacing(0)

        # Title
        title = QLabel("🚦 TBRGS");  title.setObjectName("appTitle")
        sub   = QLabel("Traffic-Based Route Guidance System")
        sub.setObjectName("appSubtitle")
        layout.addWidget(title)
        layout.addWidget(sub)
        layout.addSpacing(12)
        layout.addWidget(_hline())
        layout.addSpacing(10)

        # SCATS site list
        ref = QLabel("SCATS NETWORK NODES");  ref.setObjectName("sectionLabel")
        layout.addWidget(ref)
        layout.addSpacing(4)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setFixedHeight(148)
        scroll.setObjectName("scatsScroll")

        inner = QWidget();  inner.setObjectName("scatsInner")
        iv = QVBoxLayout(inner)
        iv.setContentsMargins(0, 0, 0, 0)
        iv.setSpacing(2)
        for nid, (_, _, lbl) in sorted(REAL_SCATS.items()):
            row = QLabel(
                f"<b style='color:#3498db'>{nid}</b>"
                f"<span style='color:#95a5a6'> — {lbl}</span>"
            )
            row.setObjectName("scatsRow")
            row.setTextFormat(Qt.RichText)
            iv.addWidget(row)
        scroll.setWidget(inner)
        layout.addWidget(scroll)
        layout.addSpacing(10)
        layout.addWidget(_hline())
        layout.addSpacing(12)

        # Route config
        grp = QGroupBox("Route Configuration")
        grp.setObjectName("configGroup")
        gl  = QVBoxLayout(grp)
        gl.setSpacing(8)

        gl.addWidget(_form_label("Origin SCATS Site"))
        self.origin_combo = QComboBox()
        self.origin_combo.setObjectName("formInput")
        self.origin_combo.setEditable(True)
        for nid in sorted(REAL_SCATS):
            self.origin_combo.addItem(str(nid))
        gl.addWidget(self.origin_combo)

        gl.addWidget(_form_label("Destination SCATS Site"))
        self.dest_combo = QComboBox()
        self.dest_combo.setObjectName("formInput")
        self.dest_combo.setEditable(True)
        for nid in sorted(REAL_SCATS):
            self.dest_combo.addItem(str(nid))
        if self.dest_combo.count() > 1:
            self.dest_combo.setCurrentIndex(1)
        gl.addWidget(self.dest_combo)

        gl.addWidget(_form_label("Prediction Model"))
        self.model_combo = QComboBox()
        self.model_combo.setObjectName("modelCombo")
        self.model_combo.addItems(["lstm", "gru", "transformer"])
        self.model_combo.setToolTip(
            "Select the trained deep-learning model to predict traffic flow.\n"
            "Falls back to 300 veh/15 min if weights are not found."
        )
        gl.addWidget(self.model_combo)

        layout.addWidget(grp)
        layout.addSpacing(12)

        # Find button
        self.find_btn = QPushButton("🔍   Find Top 5 Routes")
        self.find_btn.setObjectName("findBtn")
        self.find_btn.setFixedHeight(46)
        self.find_btn.setCursor(Qt.PointingHandCursor)
        self.find_btn.clicked.connect(self._on_find_clicked)
        layout.addWidget(self.find_btn)
        layout.addSpacing(14)

        # Results
        res_lbl = QLabel("ROUTE RESULTS");  res_lbl.setObjectName("sectionLabel")
        layout.addWidget(res_lbl)
        layout.addSpacing(4)

        self.results_text = QTextEdit()
        self.results_text.setObjectName("resultsText")
        self.results_text.setReadOnly(True)
        self.results_text.setPlaceholderText(
            "Route results will appear here.\n\n"
            "Select an origin, destination, and model,\n"
            "then click 'Find Top 5 Routes'."
        )
        layout.addWidget(self.results_text, stretch=1)
        layout.addSpacing(8)

        self.status_lbl = QLabel("Ready.")
        self.status_lbl.setObjectName("statusLabel")
        layout.addWidget(self.status_lbl)

        return panel

    def _build_map_panel(self) -> QWidget:
        container = QWidget();  container.setObjectName("mapContainer")
        vl = QVBoxLayout(container)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)

        bar = QFrame();  bar.setObjectName("mapBar");  bar.setFixedHeight(34)
        bl  = QHBoxLayout(bar)
        bl.setContentsMargins(14, 0, 14, 0)
        lbl = QLabel("🗺  OpenStreetMap — Route Visualisation")
        lbl.setObjectName("mapBarLabel")
        bl.addWidget(lbl)
        bl.addStretch()
        hint = QLabel("Click a route on the map for details")
        hint.setObjectName("mapBarHint")
        bl.addWidget(hint)
        vl.addWidget(bar)

        self.map_view = QWebEngineView()
        self.map_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        vl.addWidget(self.map_view, stretch=1)

        return container

    # ------------------------------------------------------------------
    # Default map (all SCATS sites, no routes)
    # ------------------------------------------------------------------

    def _show_default_map(self) -> None:
        fmap = folium.Map(
            location=[-37.858, 145.070], zoom_start=13, tiles="OpenStreetMap"
        )
        for nid, (lat, lon, lbl) in REAL_SCATS.items():
            html = f'''
            <div style="background-color:#3498db; color:white; font-weight:bold; font-size:12px;
                        font-family:Arial; text-align:center; line-height:22px; width:28px;
                        height:28px; border-radius:50%; border:2px solid white;
                        box-shadow:0 1px 3px rgba(0,0,0,0.3);">
                {nid}
            </div>
            '''
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(f"<b>SCATS {nid}</b><br>{lbl}", max_width=220),
                tooltip=f"SCATS {nid} — {lbl}",
                icon=folium.DivIcon(html=html, icon_size=(28, 28), icon_anchor=(14, 14))
            ).add_to(fmap)

        fmap.get_root().html.add_child(folium.Element("""
        <div style="position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);
                    z-index:9999;background:rgba(255,255,255,0.92);
                    padding:18px 28px;border-radius:12px;text-align:center;
                    box-shadow:0 4px 16px rgba(0,0,0,0.15);
                    font-family:'Segoe UI',Arial,sans-serif;">
          <div style="font-size:28px">🚦</div>
          <div style="font-size:16px;font-weight:bold;margin-top:6px">TBRGS Route Map</div>
          <div style="font-size:13px;color:#666;margin-top:4px">
            Select origin &amp; destination, then click<br>
            <b>Find Top 5 Routes</b> to visualise routes.
          </div>
        </div>
        """))
        fmap.save(MAP_OUTPUT)
        self.map_view.setUrl(QUrl.fromLocalFile(os.path.realpath(MAP_OUTPUT)))

    # ------------------------------------------------------------------
    # Slot: Find Routes button
    # ------------------------------------------------------------------

    def _on_find_clicked(self) -> None:
        try:
            origin = int(self.origin_combo.currentText().strip())
            dest   = int(self.dest_combo.currentText().strip())
        except ValueError:
            QMessageBox.warning(
                self, "Invalid Input",
                "Origin and Destination must be integer SCATS numbers."
            )
            return

        if origin == dest:
            QMessageBox.warning(
                self, "Invalid Input",
                "Origin and Destination must be different sites."
            )
            return

        model = self.model_combo.currentText().strip().lower()

        self.find_btn.setEnabled(False)
        self.results_text.clear()
        self._set_status("⏳  Computing routes …", "#f39c12")

        self._worker = RoutingWorker(origin, dest, model)
        self._worker.finished.connect(self._on_routes_ready)
        self._worker.error.connect(self._on_routing_error)
        self._worker.start()

    # ------------------------------------------------------------------
    # Slot: Routes ready
    # ------------------------------------------------------------------

    def _on_routes_ready(self, routes: list) -> None:
        self.find_btn.setEnabled(True)

        origin = int(self.origin_combo.currentText().strip())
        dest   = int(self.dest_combo.currentText().strip())
        model  = self.model_combo.currentText()

        origin_lbl = REAL_SCATS.get(origin, ("", "", str(origin)))[2]
        dest_lbl   = REAL_SCATS.get(dest,   ("", "", str(dest)))[2]

        DOTS = {"red": "🔴", "blue": "🔵", "green": "🟢",
                "purple": "🟣", "orange": "🟠"}

        lines = [
            f"  Origin      : SCATS {origin}",
            f"                {origin_lbl}",
            f"  Destination : SCATS {dest}",
            f"                {dest_lbl}",
            f"  Model       : {model}",
            f"  Routes found: {len(routes)}",
            "",
            "  ─────────────────────────────────────",
        ]
        for i, (path, tt) in enumerate(routes):
            col  = ROUTE_COLORS[i % len(ROUTE_COLORS)]
            dot  = DOTS.get(col, "●")
            lines += [
                "",
                f"  {dot}  Route {i+1}  [{col.upper()}]",
                f"     Path : {' → '.join(map(str, path))}",
                f"     Hops : {len(path) - 1}",
                f"     Time : {tt:.2f} min",
            ]
        self.results_text.setPlainText("\n".join(lines))

        self._set_status("🗺  Rendering map …", "#3498db")
        try:
            path_html = generate_map(routes, origin, dest)
            self.map_view.setUrl(QUrl.fromLocalFile(path_html))
            self._set_status(
                f"✅  {len(routes)} route(s) found — map updated.", "#27ae60"
            )
        except Exception as exc:
            self._set_status(f"Map render error: {exc}", "#e74c3c")

    # ------------------------------------------------------------------
    # Slot: Error
    # ------------------------------------------------------------------

    def _on_routing_error(self, msg: str) -> None:
        self.find_btn.setEnabled(True)
        self._set_status("❌  Routing failed.", "#e74c3c")
        QMessageBox.critical(self, "Routing Error", msg)

    # ------------------------------------------------------------------

    def _set_status(self, text: str, color: str = "#7f8c8d") -> None:
        self.status_lbl.setText(text)
        self.status_lbl.setStyleSheet(f"color:{color};font-size:11px;")


# ===========================================================================
# Widget helpers
# ===========================================================================

def _hline() -> QFrame:
    line = QFrame(); line.setFrameShape(QFrame.HLine); line.setObjectName("hLine")
    return line

def _form_label(text: str) -> QLabel:
    lbl = QLabel(text); lbl.setObjectName("formLabel"); return lbl


# ===========================================================================
# Stylesheet
# ===========================================================================

_STYLESHEET = """
QMainWindow, QWidget      { background:#f0f2f5; font-family:'Segoe UI',Arial,sans-serif; }
QScrollBar:vertical       { width:6px; background:transparent; }
QScrollBar::handle:vertical { background:#4a6080; border-radius:3px; min-height:20px; }

#leftPanel                { background:#1c2b3a; }
#appTitle                 { font-size:28px; font-weight:bold; color:#3498db; }
#appSubtitle              { font-size:11px; color:#7f8c8d; }
#hLine                    { background:#2c3e50; border:none; max-height:1px; }
#sectionLabel             { font-size:10px; font-weight:bold; color:#7f8c8d; letter-spacing:1.5px; }
#scatsScroll              { background:transparent; border:none; }
#scatsInner               { background:transparent; }
#scatsRow                 { font-size:11px; padding:1px 0; color:#bdc3c7; }

#configGroup              { border:1px solid #2c3e50; border-radius:8px; margin-top:6px;
                             padding-top:8px; color:#ecf0f1; font-size:13px; font-weight:bold; }
#configGroup::title       { subcontrol-origin:margin; left:10px; color:#3498db; padding:0 4px; }
#formLabel                { font-size:12px; color:#bdc3c7; margin-top:4px; }
#formInput, QComboBox     { background:#243447; color:#ecf0f1; border:1px solid #2c3e50;
                             border-radius:5px; padding:7px 10px; font-size:13px; }
#formInput:focus          { border:1px solid #3498db; }
QComboBox::drop-down      { border:none; width:20px; }
QComboBox QAbstractItemView { background:#243447; color:#ecf0f1;
                               selection-background-color:#3498db; border:none; }

#findBtn                  { background:#3498db; color:white; font-size:14px;
                             font-weight:bold; border:none; border-radius:8px; }
#findBtn:hover            { background:#2980b9; }
#findBtn:pressed          { background:#1a6fa1; }
#findBtn:disabled         { background:#2c3e50; color:#4a6080; }

#resultsText              { background:#172535; color:#ecf0f1; border:none;
                             border-radius:6px; font-family:'Courier New',monospace;
                             font-size:12px; padding:8px; }
#statusLabel              { font-size:11px; color:#7f8c8d; }

#mapContainer             { background:#f0f2f5; }
#mapBar                   { background:#2c3e50; }
#mapBarLabel              { color:#ecf0f1; font-size:13px; font-weight:bold; }
#mapBarHint               { color:#7f8c8d; font-size:11px; font-style:italic; }

QMessageBox               { background:#1c2b3a; color:#ecf0f1; }
QMessageBox QPushButton   { background:#3498db; color:white; border-radius:4px;
                             padding:6px 16px; font-weight:bold; }
"""


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    window = TBRGSWindow()
    window.show()
    sys.exit(app.exec_())
