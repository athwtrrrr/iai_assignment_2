"""
search.py
=========
Search algorithms for the TBRGS routing engine.

Implementations
---------------
  dijkstra()       — classic Dijkstra shortest path on a weighted adj-dict
  a_star()         — A* with admissible Haversine heuristic
  yen_k_shortest() — Yen's K Loopless Shortest Paths (calls Dijkstra)

Note on algorithm choice
------------------------
Yen's algorithm internally needs to run Dijkstra on repeatedly *pruned*
subgraphs (nodes / edges removed dynamically for each spur step).  A*
with a global heuristic is used for the *initial* shortest path (faster
on large graphs) while the spur steps use plain Dijkstra for correctness
on arbitrary pruned topologies.
"""
import math
import heapq
import itertools
from typing import Dict, List, Optional, Set, Tuple

from search_graph import TrafficGraph, haversine


# ===========================================================================
# Dijkstra's shortest path
# ===========================================================================

def dijkstra(
    weighted_adj: Dict[int, List[Tuple[int, float]]],
    source:       int,
    target:       int,
) -> Tuple[Optional[List[int]], float]:
    """
    Dijkstra's single-source shortest-path algorithm.

    Operates on a *pre-materialised* weighted adjacency dictionary so
    that the caller (Yen's spur step) can pass in a pruned sub-graph
    without modifying the main graph object.

    Parameters
    ----------
    weighted_adj : {from_id: [(to_id, cost), ...]}
    source       : starting node id
    target       : destination node id

    Returns
    -------
    (path, cost) — where path is a list of node ids from source to target.
    (None, inf)  — when target is unreachable from source.
    """
    if source == target:
        return [source], 0.0

    dist: Dict[int, float] = {source: 0.0}
    prev: Dict[int, int]   = {}
    heap = [(0.0, source)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, math.inf):
            continue                  # stale entry — skip

        if u == target:
            # Reconstruct path by walking back through prev pointers
            path: List[int] = []
            node = target
            while node in prev:
                path.append(node)
                node = prev[node]
            path.append(source)
            path.reverse()
            return path, d

        for v, w in weighted_adj.get(u, []):
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    return None, math.inf


# ===========================================================================
# A* with Haversine heuristic
# ===========================================================================

def a_star(
    graph:        TrafficGraph,
    weighted_adj: Dict[int, List[Tuple[int, float]]],
    source:       int,
    target:       int,
) -> Tuple[Optional[List[int]], float]:
    """
    A* search using the Haversine distance to *target* as the heuristic.

    Heuristic: h(n) = haversine(n, target) / SPEED_LIMIT × 60  [minutes]

    This is *admissible* (never overestimates remaining travel time)
    because Haversine gives the straight-line lower bound on distance
    and SPEED_LIMIT is the maximum possible speed.

    Parameters
    ----------
    graph        : TrafficGraph (needed to look up lat/lon for heuristic)
    weighted_adj : {from_id: [(to_id, travel_time_minutes), ...]}
    source       : starting node id
    target       : destination node id

    Returns
    -------
    (path, cost) or (None, inf) — same contract as dijkstra().
    """
    SPEED_LIMIT_KMH = 60.0

    if source == target:
        return [source], 0.0

    t_lat, t_lon = graph.get_lat_lon(target)

    def heuristic(node_id: int) -> float:
        """Admissible lower bound on remaining travel time (minutes)."""
        n_lat, n_lon = graph.get_lat_lon(node_id)
        dist_km = haversine(n_lat, n_lon, t_lat, t_lon)
        return (dist_km / SPEED_LIMIT_KMH) * 60.0

    # g_cost[n] = best known cost from source to n
    g_cost: Dict[int, float] = {source: 0.0}
    prev:   Dict[int, int]   = {}
    counter = itertools.count()
    # heap stores (f_score, tie_breaker, node_id)
    heap = [(heuristic(source), next(counter), source)]

    while heap:
        f, _, u = heapq.heappop(heap)
        g = g_cost.get(u, math.inf)

        # Discard stale heap entries
        if f > g + heuristic(u) + 1e-9:
            continue

        if u == target:
            path: List[int] = []
            node = target
            while node in prev:
                path.append(node)
                node = prev[node]
            path.append(source)
            path.reverse()
            return path, g

        for v, w in weighted_adj.get(u, []):
            new_g = g + w
            if new_g < g_cost.get(v, math.inf):
                g_cost[v] = new_g
                prev[v]   = u
                f_val = new_g + heuristic(v)
                heapq.heappush(heap, (f_val, next(counter), v))

    return None, math.inf


# ===========================================================================
# Yen's K Loopless Shortest Paths
# ===========================================================================

def yen_k_shortest(
    graph:        TrafficGraph,
    weighted_adj: Dict[int, List[Tuple[int, float]]],
    source:       int,
    target:       int,
    k:            int = 5,
) -> List[Tuple[List[int], float]]:
    """
    Yen's K Loopless Shortest Paths algorithm (Yen, 1971).

    Finds the k shortest *simple* (loop-free) paths from source to
    target in order of increasing total cost.

    Algorithm outline
    -----------------
    A[0]  ← shortest path via Dijkstra.

    For i = 1 … K−1:
      For each *spur node* spur (every node along A[i−1] except last):
        root = A[i−1][:spur_index+1]   (prefix of A[i−1] up to spur)

        Remove edges: any edge (spur → next) shared by a confirmed path
          that has the same root prefix — prevents duplication.

        Remove nodes: all nodes in root except spur itself — prevents
          loops back through the root.

        Run Dijkstra from spur → target on the pruned graph.
        If a spur path exists → candidate = root[:-1] + spur_path.
        Push candidate onto a min-heap B.

      Pop the best candidate from B → A[i].

    The removal of root nodes ensures loop-freeness; the removal of
    previously-used edges ensures diversity across returned paths.

    Parameters
    ----------
    graph        : TrafficGraph (topology reference)
    weighted_adj : pre-built {from: [(to, travel_time_min)]} dict
    source       : origin SCATS site number
    target       : destination SCATS site number
    k            : number of shortest paths requested (default 5)

    Returns
    -------
    List of (path, total_cost) tuples, length ≤ k, sorted ascending
    by total travel time.  Empty list when source is unreachable from
    target.
    """
    # ── Step 0: Initial shortest path ────────────────────────────────────
    first_path, first_cost = a_star(graph, weighted_adj, source, target)
    if first_path is None:
        return []

    # A[k] = (cost, path)  — confirmed k-th shortest paths
    A: List[Tuple[float, List[int]]] = [(first_cost, first_path)]

    # B = min-heap of candidate paths, with tie-breaker counter
    _tie = itertools.count()
    B: list = []

    # ── Steps 1 … K-1 ────────────────────────────────────────────────────
    for ki in range(1, k):
        prev_cost, prev_path = A[ki - 1]

        for i in range(len(prev_path) - 1):
            spur_node = prev_path[i]
            root_path = prev_path[: i + 1]

            # ── Edges to remove ──────────────────────────────────────────
            # For every confirmed path that shares root_path as a prefix,
            # remove the edge that leaves spur_node in that path.
            # This prevents the algorithm from generating duplicate paths.
            removed_edges: Set[Tuple[int, int]] = set()
            for _, confirmed in A:
                if (
                    len(confirmed) > i
                    and confirmed[: i + 1] == root_path
                ):
                    removed_edges.add((confirmed[i], confirmed[i + 1]))

            # ── Nodes to remove ──────────────────────────────────────────
            # Remove all root nodes except the spur node itself.
            # This prevents the spur path from looping back through
            # the root prefix (guarantees loop-freeness).
            removed_nodes: Set[int] = set(root_path[:-1])

            # ── Build pruned adjacency dict ───────────────────────────────
            pruned: Dict[int, List[Tuple[int, float]]] = {}
            for fn, neighbours in weighted_adj.items():
                if fn in removed_nodes:
                    continue
                pruned[fn] = [
                    (tn, w)
                    for tn, w in neighbours
                    if tn not in removed_nodes
                    and (fn, tn) not in removed_edges
                ]

            # ── Run Dijkstra from spur_node → target on pruned graph ──────
            spur_path, spur_cost = dijkstra(pruned, spur_node, target)
            if spur_path is None:
                continue

            # ── Compute root-path cost from the original weighted adj ──────
            root_cost = 0.0
            for j in range(len(root_path) - 1):
                u, v = root_path[j], root_path[j + 1]
                for nb, tt in weighted_adj.get(u, []):
                    if nb == v:
                        root_cost += tt
                        break

            # ── Assemble candidate = root(minus dup spur) + spur_path ─────
            candidate_path = root_path[:-1] + spur_path
            candidate_cost = root_cost + spur_cost

            # Skip if this exact path is already in the candidate heap
            if not any(p == candidate_path for _, _, p in B):
                heapq.heappush(
                    B, (candidate_cost, next(_tie), candidate_path)
                )

        if not B:
            break   # no more distinct loopless paths exist

        best_cost, _, best_path = heapq.heappop(B)
        A.append((best_cost, best_path))

    return [(path, cost) for cost, path in A]
