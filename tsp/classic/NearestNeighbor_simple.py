"""Nearest Neighbor heuristic for TSP: a simple constructive algorithm."""

import math
from typing import List, Tuple, Optional


def solve_tsp_nn(coords: List[Tuple[float, float]], start: int = 0) -> List[int]:
    """Nearest Neighbor construction heuristic.

    Starts at `start` node, repeatedly visits the nearest unvisited node.
    """
    n = len(coords)
    unvisited = set(range(n))
    tour = [start]
    unvisited.remove(start)
    current = start

    while unvisited:
        # Find nearest unvisited node
        best = min(unvisited, key=lambda j: math.dist(coords[current], coords[j]))
        tour.append(best)
        unvisited.remove(best)
        current = best

    return tour


def tour_distance(tour: List[int], coords: List[Tuple[float, float]]) -> float:
    """Compute total tour distance."""
    return sum(
        math.dist(coords[tour[i]], coords[tour[(i+1) % len(tour)]])
        for i in range(len(tour))
    )
