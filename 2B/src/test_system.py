"""
test_system.py
==============
Pytest test suite for the TBRGS system (2B/src/).
20 test cases covering all major components.

Groups
------
  A — travel_time.py     flow → speed formula           (tests 01–05)
  B — search_graph.py    graph construction & haversine  (tests 06–09)
  C — search.py          Dijkstra, A*, Yen's K-Shortest  (tests 10–13)
  D — routing.py         travel time integration          (tests 14–15)
  E — data pipeline      CSV splits & schema             (tests 16–17)
  F — edge / boundary    isolated nodes, same O=D        (tests 18–20)

Run
---
    pytest test_system.py -v
"""

import os
import sys
import math
import pytest
import pandas as pd

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKSPACE)

from travel_time  import flow_to_speed, travel_time, SPEED_LIMIT, CAPACITY_SPEED
from search_graph import TrafficGraph, haversine, build_graph
from search       import dijkstra, a_star, yen_k_shortest
from routing      import calculate_travel_time, build_travel_time_adj

_TOL = 1e-5


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def simple_graph():
    """
    4-node directed graph providing at least 2 loopless paths
    from node 1 to node 4:
        1 → 2 → 3 → 4   (cost 3.0 km)
        1 → 3 → 4        (cost 3.5 km)
    """
    g = TrafficGraph()
    for nid, lat, lon in [
        (1, -37.80, 145.00),
        (2, -37.81, 145.01),
        (3, -37.82, 145.01),
        (4, -37.83, 145.00),
    ]:
        g.add_node(nid, lat, lon, f"Site {nid}")

    for u, v, d in [
        (1, 2, 1.0), (2, 1, 1.0),
        (2, 3, 1.0), (3, 2, 1.0),
        (1, 3, 2.5), (3, 1, 2.5),
        (3, 4, 1.0), (4, 3, 1.0),
        (2, 4, 2.0), (4, 2, 2.0),
    ]:
        g.add_edge(u, v, d)
    return g


@pytest.fixture
def free_flow(simple_graph):
    """Low flow (100 veh/15 min) at every node → free-flow conditions."""
    return {nid: 100.0 for nid in simple_graph.nodes}


@pytest.fixture
def congested_flow(simple_graph):
    """High flow (500 veh/15 min) at every node → congested conditions."""
    return {nid: 500.0 for nid in simple_graph.nodes}


@pytest.fixture
def adj_free(simple_graph, free_flow):
    return build_travel_time_adj(simple_graph, free_flow)


@pytest.fixture
def adj_congested(simple_graph, congested_flow):
    return build_travel_time_adj(simple_graph, congested_flow)


# ===========================================================================
# Group A — travel_time.py  (tests 01–05)
# ===========================================================================

def test_01_zero_flow_speed_capped_at_limit():
    """Flow = 0 → empty road → speed must equal the speed limit (60 km/h)."""
    assert abs(flow_to_speed(0) - SPEED_LIMIT) < _TOL


def test_02_flow_at_boundary_still_speed_limit():
    """
    Flow = 351 veh/hr is the exact point where the parabola meets the
    speed-limit line.  Speed must be capped at 60 km/h (≤ threshold rule).
    """
    speed = flow_to_speed(351)
    assert abs(speed - SPEED_LIMIT) < 0.1, f"Expected ~60.0, got {speed:.2f}"


def test_03_flow_above_threshold_uses_quadratic():
    """
    Flow = 800 veh/hr > 351 → quadratic green-branch formula applies.
    Expected speed ≈ 53.86 km/h (regression anchor).
    """
    speed = flow_to_speed(800)
    assert 50.0 < speed < 58.0, f"Expected ~53.86 km/h, got {speed:.2f}"


def test_04_flow_at_capacity_equals_capacity_speed():
    """Flow = 1500 veh/hr (capacity turning point) → speed = 32 km/h."""
    speed = flow_to_speed(1500)
    assert abs(speed - CAPACITY_SPEED) < 0.1, f"Expected 32.0, got {speed:.2f}"


def test_05_travel_time_monotone_increasing_with_flow():
    """
    Higher flow → lower speed → longer travel time.
    Verified across the full range: free-flow through over-capacity.
    """
    flows = [0, 200, 351, 600, 1000, 1500]
    times = [travel_time(f, 1.0) for f in flows]
    for i in range(1, len(times)):
        assert times[i] >= times[i - 1], (
            f"Monotone violated: t({flows[i]}) = {times[i]:.4f} "
            f"< t({flows[i-1]}) = {times[i-1]:.4f}"
        )


# ===========================================================================
# Group B — search_graph.py  (tests 06–09)
# ===========================================================================

def test_06_haversine_same_point_returns_zero():
    """Haversine distance from a point to itself must be 0 km."""
    assert haversine(-37.8, 145.0, -37.8, 145.0) < _TOL


def test_07_haversine_known_distance():
    """
    Melbourne CBD to Hawthorn is approx 5–8 km by straight line.
    Checks that the Haversine implementation returns a plausible value.
    """
    dist = haversine(-37.8136, 144.9631, -37.8225, 145.0282)
    assert 4.0 < dist < 9.0, f"Expected 5–8 km, got {dist:.2f} km"


def test_08_traffic_graph_stores_nodes_and_edges(simple_graph):
    """TrafficGraph must correctly store nodes and directed edges."""
    assert len(simple_graph.nodes) == 4
    total_edges = sum(len(v) for v in simple_graph.adj.values())
    assert total_edges > 0, "Graph has no edges"


def test_09_build_graph_from_site_info():
    """
    build_graph() from the real site_info.csv must produce a non-empty
    TrafficGraph with at least one edge per node.
    """
    path = os.path.join(WORKSPACE, "data", "site_info.csv")
    if not os.path.exists(path):
        pytest.skip("data/site_info.csv not found — run data_processing.py first")
    g = build_graph(path, max_neighbours=2)
    assert len(g.nodes) >= 1, "Graph has no nodes"
    assert sum(len(v) for v in g.adj.values()) >= 1, "Graph has no edges"


# ===========================================================================
# Group C — search.py  (tests 10–13)
# ===========================================================================

def test_10_dijkstra_finds_valid_path(adj_free):
    """Dijkstra must return a connected path from node 1 to node 4."""
    path, cost = dijkstra(adj_free, 1, 4)
    assert path is not None, "Dijkstra returned None for a reachable target"
    assert path[0] == 1 and path[-1] == 4
    assert cost > 0


def test_11_dijkstra_unreachable_returns_none():
    """Dijkstra must return (None, inf) when the target is unreachable."""
    isolated_adj = {1: [(2, 1.0)], 2: [], 99: []}
    path, cost = dijkstra(isolated_adj, 1, 99)
    assert path is None
    assert cost == math.inf


def test_12_yen_paths_sorted_ascending(simple_graph, adj_free):
    """Yen's K paths must be returned in non-decreasing order of cost."""
    routes = yen_k_shortest(simple_graph, adj_free, 1, 4, k=5)
    assert len(routes) >= 1, "Yen's returned no routes"
    times = [tt for _, tt in routes]
    assert times == sorted(times), f"Routes not sorted: {times}"


def test_13_yen_paths_correct_endpoints(simple_graph, adj_free):
    """Every path returned by Yen's must start at origin and end at destination."""
    routes = yen_k_shortest(simple_graph, adj_free, 1, 4, k=3)
    for path, _ in routes:
        assert path[0] == 1, f"Path does not start at origin: {path}"
        assert path[-1] == 4, f"Path does not end at destination: {path}"


# ===========================================================================
# Group D — routing.py  (tests 14–15)
# ===========================================================================

def test_14_calculate_travel_time_free_flow_1km():
    """
    calculate_travel_time(flow_15min=0, distance_km=1.0) must produce
    exactly 1.5 minutes:
        flow_hr = 0 × 4 = 0  →  speed = 60 km/h
        time    = (1/60) × 60 + 0.5 = 1.5 min
    """
    tt = calculate_travel_time(0.0, 1.0)
    assert abs(tt - 1.5) < _TOL, f"Expected 1.5 min, got {tt:.4f}"


def test_15_congestion_increases_route_travel_time(simple_graph, free_flow, congested_flow):
    """
    Shortest-path travel time under congested flow must strictly exceed
    the free-flow travel time on the same graph.
    """
    adj_f = build_travel_time_adj(simple_graph, free_flow)
    adj_c = build_travel_time_adj(simple_graph, congested_flow)
    _, tt_free = dijkstra(adj_f, 1, 4)
    _, tt_cong = dijkstra(adj_c, 1, 4)
    assert tt_cong > tt_free, (
        f"Congested time ({tt_cong:.4f}) should exceed "
        f"free-flow ({tt_free:.4f})"
    )


# ===========================================================================
# Group E — data pipeline  (tests 16–17)
# ===========================================================================

def test_16_train_val_test_split_dates():
    """
    Verify the temporal train/val/test split boundaries:
        train  : timestamps < 2006-10-22
        val    : 2006-10-22 ≤ timestamps < 2006-10-27
        test   : timestamps ≥ 2006-10-27
    """
    base = os.path.join(WORKSPACE, "data")
    for name in ("train.csv", "val.csv", "test.csv"):
        if not os.path.exists(os.path.join(base, name)):
            pytest.skip(f"data/{name} not found — run data_processing.py first")

    train = pd.read_csv(os.path.join(base, "train.csv"))
    val   = pd.read_csv(os.path.join(base, "val.csv"))
    test  = pd.read_csv(os.path.join(base, "test.csv"))

    for df in (train, val, test):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    assert train["timestamp"].max() < pd.Timestamp("2006-10-22"), "Train overflows into val"
    assert val["timestamp"].min()  >= pd.Timestamp("2006-10-22"), "Val starts too early"
    assert val["timestamp"].max()  <  pd.Timestamp("2006-10-27"), "Val overflows into test"
    assert test["timestamp"].min() >= pd.Timestamp("2006-10-27"), "Test starts too early"


def test_17_site_info_required_columns():
    """site_info.csv must contain the four required columns."""
    path = os.path.join(WORKSPACE, "data", "site_info.csv")
    if not os.path.exists(path):
        pytest.skip("data/site_info.csv not found")
    df = pd.read_csv(path)
    for col in ("scats_id", "lat", "lon", "location"):
        assert col in df.columns, f"Missing required column: '{col}'"
    assert len(df) > 0, "site_info.csv is empty"


# ===========================================================================
# Group F — edge / boundary cases  (tests 18–20)
# ===========================================================================

def test_18_dijkstra_same_origin_destination(adj_free):
    """When source == target, Dijkstra must return the trivial path at cost 0."""
    path, cost = dijkstra(adj_free, 2, 2)
    assert path == [2], f"Expected [2], got {path}"
    assert cost == 0.0, f"Expected cost 0.0, got {cost}"


def test_19_yen_isolated_node_returns_empty_list(simple_graph, free_flow):
    """
    Yen's K-Shortest Paths to an isolated node (no incoming edges)
    must return an empty list rather than raising an exception.
    """
    simple_graph.add_node(99, -39.0, 147.0, "Isolated Site")
    free_flow[99] = 100.0
    adj = build_travel_time_adj(simple_graph, free_flow)
    routes = yen_k_shortest(simple_graph, adj, 1, 99, k=5)
    assert routes == [], f"Expected [], got {routes}"


def test_20_flow_to_speed_always_positive():
    """
    flow_to_speed must never return zero or negative speed for any
    non-negative flow value, including extreme inputs.
    """
    for flow in [0, 100, 351, 800, 1500, 2000, 10000]:
        speed = flow_to_speed(float(flow))
        assert speed > 0, f"Got non-positive speed {speed} for flow={flow}"
