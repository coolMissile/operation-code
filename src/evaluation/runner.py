"""Evaluation runner: execute candidate code on TSP instances and measure performance."""

import time
import math
from typing import Dict, Any, Optional, List

from src.core import TSPInstance, TSPSolution
from src.io.problem_reader import ProblemReader


def eval_code(code_str: str, instance: TSPInstance) -> Dict[str, Any]:
    """Evaluate a code snippet on a single TSP instance.

    Expects the code to define a `solve()` function that returns either:
    - A TSPSolution object
    - A list[int] (tour permutation)
    - A dict with 'tour' and 'distance' keys
    """
    namespace = {
        "math": math,
        "instance": instance,
        "nodes": instance.nodes,
        "dimension": instance.dimension,
    }

    try:
        exec(compile(code_str, "<eval>", "exec"), namespace)
    except Exception as e:
        return {"success": False, "error": f"Compile failed: {e}"}

    solver = namespace.get("solve") or namespace.get("tsp_solve")
    if not solver:
        return {"success": False, "error": "No solve() or tsp_solve() function found"}

    start = time.time()
    try:
        result = solver()
        runtime = time.time() - start

        if isinstance(result, TSPSolution):
            sol = result
        elif isinstance(result, list):
            dist = sum(
                math.dist(instance.nodes[result[i]],
                          instance.nodes[result[(i + 1) % len(result)]])
                for i in range(len(result))
            )
            sol = TSPSolution(tour=result, distance=dist, method="evolved", runtime=runtime)
        elif isinstance(result, dict):
            sol = TSPSolution(
                tour=result.get("tour", []),
                distance=result.get("distance", float("inf")),
                method="evolved", runtime=runtime,
            )
        else:
            return {"success": False, "error": f"Unexpected return type: {type(result)}"}
    except Exception as e:
        return {"success": False, "error": f"Runtime error: {e}"}

    valid = sol.validate(instance)
    gap_val = sol.gap(instance)
    return {
        "success": valid,
        "valid": valid,
        "distance": sol.distance,
        "gap": gap_val,
        "runtime": sol.runtime,
        "method": sol.method,
        "tour": sol.tour,
    }


class BenchmarkRunner:
    """Run candidate code on benchmark instances."""

    def __init__(self, tsplib_dir: str = "data/tsplib"):
        self.reader = ProblemReader()
        self.tsplib_dir = tsplib_dir

    def run(self, code_str: str,
            instance_names: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Evaluate code on multiple instances."""
        if instance_names is None:
            instance_names = ["eil51", "berlin52", "kroA100"]

        results = {}
        for name in instance_names:
            inst = self.reader.load_tsplib_dir(self.tsplib_dir, [name]).get(name)
            if inst is None:
                results[name] = {"success": False, "error": f"Instance {name} not found"}
            else:
                results[name] = eval_code(code_str, inst)
        return results

    def run_on_random(self, code_str: str, sizes: List[int] = None) -> Dict[str, Dict]:
        """Evaluate on random instances for quick testing."""
        if sizes is None:
            sizes = [10, 20, 50]
        results = {}
        for n in sizes:
            inst = self.reader.random_instance(f"random_{n}", n, seed=42)
            results[inst.name] = eval_code(code_str, inst)
        return results
