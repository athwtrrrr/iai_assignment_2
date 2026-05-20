"""
router.py – Route finding module for TBRGS.

Builds a directed graph with travel-time edge weights derived from
ML-predicted traffic flow, then returns the top-k shortest routes
using Yen's k-shortest-simple-paths algorithm (via NetworkX).
"""

import os
import sys
from math import sqrt

import networkx as nx

# ── ensure local imports resolve when called from GUI ──────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from graph import build_graph
from predict import predict_flow

# ── physics constants (must match travel_time.py) ──────────────────
_CAPACITY_FLOW  = 1500   # vehicles/hour at capacity
_CAPACITY_SPEED = 32     # km/h at capacity
_A = -_CAPACITY_FLOW / (_CAPACITY_SPEED ** 2)
_B = -2 * _CAPACITY_SPEED * _A

# ── module-level cache ──────────────────────────────────────────────
_base_graph = None
_base_sites = None


def _get_base_graph():
    global _base_graph, _base_sites
    if _base_graph is None:
        _base_graph, _base_sites = build_graph()
    return _base_graph, _base_sites


def _flow_to_speed(flow_per_hour: float, speed_limit: float) -> float:
    """Convert hourly flow (veh/h) to speed (km/h), capped at speed_limit."""
    delta = _B ** 2 - 4 * _A * flow_per_hour
    if delta < 0:
        return 1.0
    sqrt_delta = sqrt(delta)
    if flow_per_hour <= _CAPACITY_FLOW:
        speed = (-_B + sqrt_delta) / (2 * _A)
    else:
        speed = (-_B - sqrt_delta) / (2 * _A)
    return max(1.0, min(speed, speed_limit))


def build_weighted_digraph(
    timestamp: str,
    model: str = "lstm",
    speed_limit: float = 60.0,
    intersection_delay: float = 30.0,
) -> nx.DiGraph:
    """
    Build a directed weighted graph where each edge weight is the
    predicted travel time (seconds) from u to v.

    Travel time formula:
        speed   = flow_to_speed(predicted_flow_at_v * 4)   [*4: 15-min → hourly]
        t_edge  = (distance_km / speed) * 3600 + intersection_delay
    """
    G, _ = _get_base_graph()

    # Pre-compute predicted hourly flow for every node
    flow_cache: dict[int, float] = {}
    for site_id in G.nodes():
        try:
            flow_15 = predict_flow(site_id, timestamp, model=model)
            flow_cache[site_id] = max(0.0, float(flow_15) * 4)
        except Exception:
            flow_cache[site_id] = 0.0  # fallback → free-flow speed

    DG = nx.DiGraph()
    for node, data in G.nodes(data=True):
        DG.add_node(node, **data)

    for u, v, data in G.edges(data=True):
        dist_km = data["distance_km"]
        for src, dst in [(u, v), (v, u)]:
            flow_ph = flow_cache.get(dst, 0.0)
            speed   = _flow_to_speed(flow_ph, speed_limit)
            t_sec   = (dist_km / speed) * 3600 + intersection_delay
            DG.add_edge(src, dst, distance_km=dist_km, travel_time=t_sec, flow_ph=flow_ph)

    return DG


def find_top_k_routes(
    origin: int,
    destination: int,
    timestamp: str,
    model: str = "lstm",
    k: int = 5,
    speed_limit: float = 60.0,
    intersection_delay: float = 30.0,
) -> list[dict]:
    """
    Find the top-k shortest routes (by travel time) from origin to destination.

    Returns a list of dicts, each with:
        path              – list of SCATS site IDs
        travel_time       – total seconds (float)
        num_intersections – number of edges traversed
    """
    DG = build_weighted_digraph(timestamp, model, speed_limit, intersection_delay)

    if origin not in DG.nodes:
        raise ValueError(f"Origin site {origin} is not in the road network.")
    if destination not in DG.nodes:
        raise ValueError(f"Destination site {destination} is not in the road network.")

    routes = []
    try:
        gen = nx.shortest_simple_paths(DG, origin, destination, weight="travel_time")
        for i, path in enumerate(gen):
            if i >= k:
                break
            total_time = sum(
                DG[path[j]][path[j + 1]]["travel_time"]
                for j in range(len(path) - 1)
            )
            routes.append({
                "path": path,
                "travel_time": total_time,
                "num_intersections": len(path) - 1,
            })
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass

    return routes


# ── CLI smoke-test ──────────────────────────────────────────────────
if __name__ == "__main__":
    os.chdir(_DIR)
    print("Finding top-5 routes: 2000 → 3002 at 2006-10-27 08:00 (LSTM)…")
    routes = find_top_k_routes(2000, 3002, "2006-10-27 08:00", model="lstm", k=5)
    if not routes:
        print("No routes found.")
    for i, r in enumerate(routes, 1):
        mins = r["travel_time"] / 60
        path = " → ".join(str(n) for n in r["path"])
        print(f"Route {i}: {mins:.1f} min — {path}")
