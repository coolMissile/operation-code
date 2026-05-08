"""
Design Space: construct and explore the design space from extracted algorithms.
"""

import itertools
from typing import List, Dict, Set, Optional
from collections import Counter

from src.core import AlgorithmDescription, DesignDimension, DesignVector, DesignSpace


class DesignSpaceConstructor:
    """Build a design space from extracted algorithm descriptions."""

    def scan_dimensions(self, algorithms: List[AlgorithmDescription]) -> DesignSpace:
        """Induce design dimensions and map known points."""
        value_map: Dict[str, tuple[str, Set[str]]] = {
            "representation": ("Solution representation", set()),
            "search_type": ("Search strategy", set()),
            "initialization": ("Initial solution generation", set()),
            "neighborhood": ("Neighborhood/search operator(s)", set()),
            "selection": ("Selection mechanism", set()),
            "acceptance": ("Acceptance criterion", set()),
            "population": ("Population management", set()),
            "termination": ("Termination condition", set()),
        }

        algo_vectors: Dict[str, DesignVector] = {}

        for algo in algorithms:
            values = {
                "representation": algo.representation.value,
                "search_type": algo.search_type.value,
                "initialization": algo.initialization,
                "neighborhood": algo.neighborhood,
                "selection": algo.selection,
                "acceptance": algo.acceptance,
                "population": algo.population,
                "termination": algo.termination,
            }
            for dim, val in values.items():
                if val:
                    value_map[dim][1].add(val)
            algo_vectors[algo.algorithm_id] = DesignVector(dimensions=values)

        dimensions = [
            DesignDimension(name=dim, possible_values=sorted(vals), description=desc)
            for dim, (desc, vals) in value_map.items()
            if vals
        ]

        return DesignSpace(
            dimensions=dimensions,
            known_points=algo_vectors,
            name="TSP Solver Design Space",
        )

    def suggest_unexplored(self, space: DesignSpace, top_k: int = 10) -> List[Dict[str, str]]:
        """Suggest unexplored design combinations."""
        all_values = [d.possible_values for d in space.dimensions]
        dim_names = [d.name for d in space.dimensions]
        known_set: Set[tuple] = set()

        for vec in space.known_points.values():
            known_set.add(tuple(sorted(
                (k, v) for k, v in vec.dimensions.items() if k in dim_names
            )))

        candidates = []
        for combo in itertools.product(*all_values):
            vec = tuple(sorted(zip(dim_names, combo)))
            if vec not in known_set:
                candidates.append(dict(vec))
                if len(candidates) >= top_k:
                    break
        return candidates

    def design_space_summary(self, space: DesignSpace) -> str:
        """Generate a structured text summary of the design space."""
        lines = [f"# {space.name}\n", "## Design Dimensions\n"]

        for dim in space.dimensions:
            lines.append(f"### {dim.name}")
            if dim.description:
                lines.append(f"_{dim.description}_\n")
            for val in dim.possible_values:
                users = [
                    aid for aid, vec in space.known_points.items()
                    if vec.dimensions.get(dim.name) == val
                ]
                user_str = f" (used by: {', '.join(users)})" if users else " (UNEXPLORED)"
                lines.append(f"- `{val}`{user_str}")
            lines.append("")

        lines.append("## Coverage\n")
        total = 1
        for dim in space.dimensions:
            total *= len(dim.possible_values)
        lines.append(f"Total theoretical combinations: {total}")
        lines.append(f"Known algorithms: {len(space.known_points)}")

        lines.append("\n## Suggested Unexplored Combinations\n")
        for i, combo in enumerate(self.suggest_unexplored(space, top_k=10), 1):
            lines.append(f"{i}. " + ", ".join(f"{k}={v}" for k, v in combo.items()))

        return "\n".join(lines)
