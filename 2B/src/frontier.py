"""
frontier.py
===========
Priority Queue data structure for A* and Dijkstra search algorithms.
Uses Python's heapq (min-heap) internally.  A monotonic counter breaks
ties so that items with equal priority are returned in insertion order.
"""
import heapq
import itertools
from typing import Any, Tuple


class PriorityQueue:
    """
    Min-heap priority queue with stable tie-breaking.

    Each item is stored internally as (priority, counter, data).
    The counter guarantees that two items with identical priority
    never trigger a comparison on the data itself (which may not
    support < / > operators).

    Usage
    -----
    pq = PriorityQueue()
    pq.push(2.5, node_a)
    pq.push(1.0, node_b)
    priority, item = pq.pop()   # → (1.0, node_b)
    """

    def __init__(self) -> None:
        self._heap: list               = []
        self._counter = itertools.count()   # monotonically increasing

    # ------------------------------------------------------------------
    def push(self, priority: float, item: Any) -> None:
        """
        Insert *item* with the given *priority*.
        Lower numeric priority ⟹ higher precedence (min-heap).
        """
        heapq.heappush(self._heap, (priority, next(self._counter), item))

    def pop(self) -> Tuple[float, Any]:
        """
        Remove and return *(priority, item)* with the lowest priority.
        Raises IndexError when the queue is empty.
        """
        priority, _, item = heapq.heappop(self._heap)
        return priority, item

    def is_empty(self) -> bool:
        return len(self._heap) == 0

    def __len__(self) -> int:
        return len(self._heap)

    def __repr__(self) -> str:
        return f"PriorityQueue(size={len(self)})"
