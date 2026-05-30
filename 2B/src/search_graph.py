"""
search_graph.py
===============
Graph data structure for the Boroondara SCATS road network.

The graph is loaded from ``data/site_info.csv`` produced by
``data_processing.py``.  Each unique SCATS site becomes a node;
directed edges connect every site to its *max_neighbours* nearest
sites (by Haversine distance) in both directions.

Coordinate correction
---------------------
VicRoads SCATS data uses the AGD84 datum while Folium / OpenStreetMap
use WGS84.  A small fixed offset aligns the markers with actual
Melbourne intersections on the map.
"""
import math
import pandas as pd
from typing import Dict, List, Optional, Tuple
from config import cfg


# Approximate AGD84 → WGS84 correction for the Boroondara / Victoria area
LAT_OFFSET = cfg["graph"]["lat_offset"]   # shift ~120 m south
LON_OFFSET = cfg["graph"]["lon_offset"]   # shift  ~90 m east


# ===========================================================================
# Node
# ===========================================================================

class Node:
    """A single SCATS intersection (vertex in the road network)."""

    def __init__(
        self,
        node_id:  int,
        lat:      float,
        lon:      float,
        location: str = "",
    ) -> None:
        self.id       = node_id
        self.lat      = lat
        self.lon      = lon
        self.location = location

    def __repr__(self) -> str:
        return f"Node({self.id}, '{self.location}')"


# ===========================================================================
# TrafficGraph
# ===========================================================================

class TrafficGraph:
    """
    Directed road network for the Boroondara SCATS area.

    Nodes  : keyed by SCATS site number (int)
    Edges  : directed, stored as (to_id, distance_km)
    Weights: distance_km is the *topology* weight; travel-time weights
             are computed on-the-fly in routing.py.
    """

    def __init__(self) -> None:
        # node_id → Node object
        self.nodes: Dict[int, Node] = {}
        # node_id → [(to_id, distance_km), ...]
        self.adj:   Dict[int, List[Tuple[int, float]]] = {}

    # ------------------------------------------------------------------
    def add_node(
        self,
        node_id:  int,
        lat:      float,
        lon:      float,
        location: str = "",
    ) -> None:
        """Register a new SCATS site as a graph node."""
        self.nodes[node_id] = Node(node_id, lat, lon, location)
        self.adj.setdefault(node_id, [])

    def add_edge(self, from_id: int, to_id: int, dist_km: float) -> None:
        """Add a directed edge from *from_id* to *to_id*."""
        self.adj.setdefault(from_id, []).append((to_id, dist_km))

    # ------------------------------------------------------------------
    def neighbors(self, node_id: int) -> List[Tuple[int, float]]:
        """Return [(neighbour_id, distance_km)] for the given node."""
        return self.adj.get(node_id, [])

    def get_lat_lon(self, node_id: int) -> Tuple[float, float]:
        """Return (latitude, longitude) for the given node."""
        node = self.nodes[node_id]
        return node.lat, node.lon

    def __repr__(self) -> str:
        n_edges = sum(len(v) for v in self.adj.values())
        return f"TrafficGraph(nodes={len(self.nodes)}, edges={n_edges})"


# ===========================================================================
# Haversine distance
# ===========================================================================

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance in kilometres between two WGS-84 coordinates.

    Uses the Haversine formula:
        a = sin²(Δlat/2) + cos(lat1)·cos(lat2)·sin²(Δlon/2)
        c = 2·arcsin(√a)
        d = R·c     where R = 6 371 km
    """
    R    = 6_371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a    = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2.0 * math.asin(math.sqrt(max(a, 0.0)))


# ===========================================================================
# Graph builder
# ===========================================================================

def build_graph(
    site_info_path: str,
    max_neighbours: int = 3,
) -> TrafficGraph:
    """
    Build a TrafficGraph from ``site_info.csv``.

    Each SCATS site becomes a node.  To form edges, every site is
    connected to its *max_neighbours* nearest sites by Haversine
    distance.  Edges are added in both directions (A→B and B→A)
    to ensure the graph is always navigable, even when the road
    network is sparse.

    Parameters
    ----------
    site_info_path  : absolute path to ``data/site_info.csv``
    max_neighbours  : number of closest sites to link per node

    Returns
    -------
    TrafficGraph with all nodes and bidirectional edges populated
    """
    df = pd.read_csv(site_info_path)

    graph = TrafficGraph()

    # ── Add nodes (apply datum correction) ───────────────────────────
    for _, row in df.iterrows():
        graph.add_node(
            int(row["scats_id"]),
            float(row["lat"])      + LAT_OFFSET,
            float(row["lon"])      + LON_OFFSET,
            str(row.get("location", "")),
        )

    # ── Add edges ────────────────────────────────────────────────────
    node_ids = list(graph.nodes.keys())

    for nid in node_ids:
        lat1, lon1 = graph.get_lat_lon(nid)

        # Sort all other sites by haversine distance from nid
        distances = sorted(
            [
                (haversine(lat1, lon1, *graph.get_lat_lon(other)), other)
                for other in node_ids
                if other != nid
            ]
        )

        seen: set = set()
        for dist, other_id in distances[:max_neighbours]:
            for u, v in [(nid, other_id), (other_id, nid)]:
                if (u, v) not in seen:
                    graph.add_edge(u, v, dist)
                    seen.add((u, v))

    return graph
