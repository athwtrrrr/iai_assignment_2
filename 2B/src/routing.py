"""
routing.py
==========
Bridge between the GUI, the Yen's K-Shortest Paths search engine,
and the trained per-site ML models.

Responsibilities
----------------
1. Load per-site model weights  (lstm_site_{id}.pth, etc.)
2. Predict traffic flow for every SCATS site.
3. Convert predicted flow (veh/15 min) → travel time (minutes)
   using the parabolic speed model from the assignment PDF.
4. Materialise a travel-time-weighted adjacency dict for the search.
5. Run Yen's K=5 to return the top routes.
"""
import os, sys
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Optional, Set, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from search_graph import TrafficGraph, build_graph
from search       import yen_k_shortest, a_star
from lstm        import LSTMModel
from gru         import GRUModel
from transformer import TransformerModel
from travel_time import travel_time   
from config import cfg

DATA_DIR   = os.path.join(SCRIPT_DIR, "data")
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FALLBACK_FLOW = cfg["routing"]["fallback_flow"]
MAX_NEIGHBOURS = cfg["routing"]["max_neighbours"]

_graph_cache   = None
_model_cache:  Dict[str, torch.nn.Module] = {}
_scaler_cache  = None
_history_cache = None

_MODEL_CLASSES = {
    "lstm":        LSTMModel,
    "gru":         GRUModel,
    "transformer": TransformerModel,
}

# ===========================================================================
# Graph
# ===========================================================================

def build_traffic_graph() -> TrafficGraph:
    global _graph_cache
    if _graph_cache is None:
        _graph_cache = build_graph(
            os.path.join(DATA_DIR, "site_info.csv"),
            max_neighbours=MAX_NEIGHBOURS,
        )
        print(f"[Routing] {_graph_cache}")
    return _graph_cache


def build_travel_time_adj(
    graph:         TrafficGraph,
    flow_map:      Dict[int, float],
    removed_edges: Optional[Set[Tuple[int, int]]] = None,
    removed_nodes: Optional[Set[int]]             = None,
) -> Dict[int, List[Tuple[int, float]]]:
    """Build travel-time weighted adj dict, honouring removal sets."""
    removed_edges = removed_edges or set()
    removed_nodes = removed_nodes or set()
    weighted: Dict[int, List[Tuple[int, float]]] = {}

    for from_id, neighbours in graph.adj.items():
        if from_id in removed_nodes:
            continue
        flow   = flow_map.get(from_id, FALLBACK_FLOW)
        bucket = []
        for to_id, dist_km in neighbours:
            if to_id in removed_nodes or (from_id, to_id) in removed_edges:
                continue
            bucket.append((to_id, travel_time(flow, dist_km)))
        weighted[from_id] = bucket

    return weighted


# ===========================================================================
# ML helpers
# ===========================================================================

def _get_scaler():
    global _scaler_cache
    if _scaler_cache is None:
        from sklearn.preprocessing import MinMaxScaler
        train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
        _scaler_cache = MinMaxScaler(feature_range=(0, 1))
        _scaler_cache.fit(train[["flow_15min"]])
    return _scaler_cache


def _get_history() -> pd.DataFrame:
    global _history_cache
    if _history_cache is None:
        parts = [
            pd.read_csv(os.path.join(DATA_DIR, f"{s}.csv"))
            for s in ("train", "val", "test")
            if os.path.exists(os.path.join(DATA_DIR, f"{s}.csv"))
        ]
        df = pd.concat(parts, ignore_index=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        _history_cache = df
    return _history_cache


def _load_model(model_name: str, site_id: int) -> Optional[torch.nn.Module]:
    """Load and cache a per-site .pth file. Returns None if not found."""
    key = f"{model_name}_{site_id}"
    if key in _model_cache:
        return _model_cache[key]
    pth = os.path.join(MODELS_DIR, f"{model_name}_site_{site_id}.pth")
    if not os.path.exists(pth):
        return None
    cls = _MODEL_CLASSES.get(model_name.lower())
    if cls is None:
        return None
    model = cls()
    model.load_state_dict(torch.load(pth, map_location=DEVICE))
    model.to(DEVICE).eval()
    _model_cache[key] = model
    return model


def predict_flows(model_name: str, lags = cfg["models"]["lags"]) -> Dict[int, float]:
    """
    Predict next-step flow (veh/15 min) for every SCATS site.
    Falls back to 300 veh/15 min when the per-site model is missing.
    """
    scaler    = _get_scaler()
    history   = _get_history()
    last_ts   = history["timestamp"].max()
    site_info = pd.read_csv(os.path.join(DATA_DIR, "site_info.csv"))
    flow_map: Dict[int, float] = {}

    for _, row in site_info.iterrows():
        sid = int(row["scats_id"])

        site_hist = (
            history[
                (history["scats_id"] == sid) &
                (history["timestamp"] <= last_ts)
            ]
            .sort_values("timestamp")
            .tail(lags)
        )
        if len(site_hist) < lags:
            flow_map[sid] = FALLBACK_FLOW
            continue

        raw    = site_hist["flow_15min"].values.reshape(-1, 1).astype("float32")
        scaled = scaler.transform(raw)

        model = _load_model(model_name.lower(), sid)
        if model is None:
            flow_map[sid] = FALLBACK_FLOW
            continue

        x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = model(x).cpu().numpy()
        flow_map[sid] = max(float(scaler.inverse_transform(pred)[0, 0]), 0.0)

    return flow_map


# ===========================================================================
# Public API
# ===========================================================================

def get_top_k_routes(
    origin:      int,
    destination: int,
    model_name:  str,
    k: int = cfg["routing"]["k_paths"],
) -> List[Tuple[List[int], float]]:
    """
    Find top-k routes by travel time between two SCATS sites.

    Returns
    -------
    [(path, total_minutes), ...] sorted ascending, length <= k.
    Empty list if no path exists.

    Raises
    ------
    ValueError  if origin or destination is unknown, or if origin == destination.
    """
    graph = build_traffic_graph()

    if origin not in graph.nodes:
        raise ValueError(
            f"Origin {origin} not in network. "
            f"Available: {sorted(graph.nodes.keys())}"
        )
    if destination not in graph.nodes:
        raise ValueError(
            f"Destination {destination} not in network. "
            f"Available: {sorted(graph.nodes.keys())}"
        )
    if origin == destination:
        raise ValueError("Origin and destination must be different.")

    print(f"[Routing] Predicting flows ({model_name}) ...")
    flow_map     = predict_flows(model_name)
    weighted_adj = build_travel_time_adj(graph, flow_map)

    print(f"[Routing] Yen K={k}: {origin} -> {destination}")
    routes = yen_k_shortest(graph, weighted_adj, origin, destination, k=k)

    for i, (path, tt) in enumerate(routes, 1):
        print(f"  Route {i}: [{' -> '.join(map(str, path))}]  {tt:.2f} min")

    return routes