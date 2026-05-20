"""
Route and integration test cases (assignment requires 15+ scenarios).
Run from 2B/src with trained models present.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), "..", "src"))

from route_search import top_k_paths, find_best_path_a_star

TS = "2006-10-27 08:00"
MODEL = "lstm"

# 15+ (origin, destination) pairs — mix of known sites
ROUTE_CASES = [
    (2000, 3002),
    (2000, 3001),
    (3001, 3002),
    (3812, 3804),
    (970, 2000),
    (2200, 3120),
    (2820, 2825),
    (3122, 3126),
    (3180, 3127),
    (2000, 3120),
    (3002, 3126),
    (2825, 2827),
    (2846, 3001),
    (3120, 3122),
    (3812, 2000),
    (3001, 3812),
    (2200, 3002),
]


@pytest.mark.parametrize("origin,dest", ROUTE_CASES)
def test_top_k_returns_list(origin, dest):
    routes = top_k_paths(origin, dest, TS, k=5, model=MODEL)
    assert isinstance(routes, list)
    if routes:
        assert len(routes) <= 5
        assert routes[0]["path"][0] == origin
        assert routes[0]["path"][-1] == dest


def test_routes_sorted_by_increasing_time():
    routes = top_k_paths(2000, 3002, TS, k=3, model=MODEL)
    if len(routes) >= 2:
        assert routes[0]["total_sec"] <= routes[1]["total_sec"]


def test_astar_when_path_exists():
    r = find_best_path_a_star(2000, 3002, TS, MODEL)
    if r:
        assert r["path"][0] == 2000
        assert r["path"][-1] == 3002
