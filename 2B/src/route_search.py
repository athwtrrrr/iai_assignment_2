"""
Top-k route search on the Boroondara SCATS network.
Integrates ML flow prediction, travel-time edges, and Part A search (A*).
"""
import importlib.util
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

_SRC = Path(__file__).resolve().parent
_A2A = _SRC.parents[1] / "2A" / "src"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Part A: import as package (src.graph / src.search)
_A2A_ROOT = _A2A.parent
if str(_A2A_ROOT) not in sys.path:
    sys.path.insert(0, str(_A2A_ROOT))
import src.graph as _a2a_graph_mod  # noqa: E402
import src.search as _a2a_search_mod  # noqa: E402

Graph = _a2a_graph_mod.Graph
Node = _a2a_graph_mod.Node
a_star_search = _a2a_search_mod.a_star_search
build_graph = _load_module("boro_graph", _SRC / "graph.py").build_graph

from predict import predict_flow
from travel_time import travel_time

FLOW_TO_HOURLY = 4  # 15-min counts -> vehicles per hour


def flow_15min_to_hourly(flow_15min):
    """Convert 15-min count to veh/hr; cap at capacity (assignment assumes under-capacity)."""
    hourly = max(0.0, flow_15min) * FLOW_TO_HOURLY
    from travel_time import CAPACITY_FLOW

    return min(hourly, CAPACITY_FLOW)


def edge_travel_time(
    site_from,
    site_to,
    distance_km,
    timestamp,
    model="lstm",
    use_predicted=True,
    history_df=None,
):
    """
    Travel time (seconds) for link site_from -> site_to.
    Assignment: use accumulated volume per hour at destination site_to.
    """
    if use_predicted:
        flow_15 = predict_flow(site_to, timestamp, model=model)
    else:
        if history_df is None:
            raise ValueError("history_df required when use_predicted=False")
        row = history_df[
            (history_df["scats_id"] == int(site_to))
            & (history_df["timestamp"] == pd.Timestamp(timestamp))
        ]
        if row.empty:
            raise ValueError(f"No flow record for site {site_to} at {timestamp}")
        flow_15 = float(row.iloc[0]["flow_15min"])

    return travel_time(flow_15min_to_hourly(flow_15), distance_km)


def build_weighted_graph(
    timestamp,
    model="lstm",
    site_info_path="data/site_info.csv",
    max_distance_km=2.0,
    use_predicted=True,
    history_df=None,
):
    """NetworkX graph with travel_time_sec on each edge."""
    G_base, _ = build_graph(site_info_path, max_distance_km)
    G = nx.DiGraph()

    for n, data in G_base.nodes(data=True):
        G.add_node(n, **data)

    for u, v, data in G_base.edges(data=True):
        dist = data["distance_km"]
        t_uv = edge_travel_time(u, v, dist, timestamp, model, use_predicted, history_df)
        t_vu = edge_travel_time(v, u, dist, timestamp, model, use_predicted, history_df)
        G.add_edge(u, v, distance_km=dist, travel_time_sec=t_uv, weight=t_uv)
        G.add_edge(v, u, distance_km=dist, travel_time_sec=t_vu, weight=t_vu)

    return G


def to_part_a_graph(nx_g: nx.DiGraph, origin: int, destination: int) -> Graph:
    """Convert weighted NetworkX graph to Part A Graph for A*."""
    g = Graph()
    for n, data in nx_g.nodes(data=True):
        g.add_node(Node(int(n), data["lon"], data["lat"]))

    for u, v, data in nx_g.edges(data=True):
        g.add_edge(int(u), int(v), float(data["weight"]))

    g.origin = int(origin)
    g.destinations = [int(destination)]
    return g


def top_k_paths(
    origin,
    destination,
    timestamp,
    k=5,
    model="lstm",
    max_distance_km=2.0,
    use_predicted=True,
):
    """
    Return up to k simple paths ordered by total travel time (seconds).
    Each item: dict with path (list of site ids), total_sec, legs (list of per-edge times).
    """
    origin, destination = int(origin), int(destination)
    G = build_weighted_graph(
        timestamp, model, max_distance_km=max_distance_km, use_predicted=use_predicted
    )

    if origin not in G or destination not in G:
        raise ValueError(f"Origin {origin} or destination {destination} not in network")

    if not nx.has_path(G, origin, destination):
        return []

    paths = []
    try:
        gen = nx.shortest_simple_paths(G, origin, destination, weight="weight")
        for path in gen:
            total = 0.0
            legs = []
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                w = G[u][v]["travel_time_sec"]
                legs.append({"from": u, "to": v, "sec": w})
                total += w
            paths.append({"path": path, "total_sec": total, "legs": legs})
            if len(paths) >= k:
                break
    except nx.NetworkXNoPath:
        pass

    return paths


def format_route_summary(routes, origin, destination):
    """Human-readable summary for CLI/GUI."""
    lines = [
        f"Routes from {origin} to {destination} (up to {len(routes)} found):",
        "",
    ]
    for i, r in enumerate(routes, 1):
        mins = r["total_sec"] / 60
        path_str = " -> ".join(str(n) for n in r["path"])
        lines.append(f"  Route {i}: {mins:.1f} min ({r['total_sec']:.0f} s)")
        lines.append(f"    {path_str}")
    if not routes:
        lines.append("  No path found between these intersections.")
    return "\n".join(lines)


def find_best_path_a_star(origin, destination, timestamp, model="lstm"):
    """Single optimal path using Part A A* on the time-weighted graph."""
    G = build_weighted_graph(timestamp, model)
    pa = to_part_a_graph(G, origin, destination)
    goal, nodes_created, path = a_star_search(pa)
    if goal is None:
        return None
    total = sum(G[path[i]][path[i + 1]]["travel_time_sec"] for i in range(len(path) - 1))
    return {
        "path": path,
        "total_sec": total,
        "nodes_expanded": nodes_created,
        "algorithm": "A* (Part A)",
    }


if __name__ == "__main__":
    ts = "2006-10-27 08:00"
    o, d = 2000, 3002
    routes = top_k_paths(o, d, ts, k=5, model="lstm")
    print(format_route_summary(routes, o, d))
    astar = find_best_path_a_star(o, d, ts)
    if astar:
        print(f"\nPart A A*: {astar['total_sec']:.0f}s, expanded {astar['nodes_expanded']} nodes")
