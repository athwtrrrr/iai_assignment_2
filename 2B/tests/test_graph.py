import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graph import build_graph, haversine


def test_haversine_same_point_zero():
    assert haversine(-37.8, 145.0, -37.8, 145.0) == 0.0


def test_graph_has_nodes():
    os.chdir(os.path.join(os.path.dirname(__file__), "..", "src"))
    G, sites = build_graph()
    assert G.number_of_nodes() == len(sites)
    assert G.number_of_edges() > 0


def test_graph_edges_have_distance():
    os.chdir(os.path.join(os.path.dirname(__file__), "..", "src"))
    G, _ = build_graph()
    u, v, d = next(iter(G.edges(data=True)))
    assert d["distance_km"] > 0
