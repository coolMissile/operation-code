"""Solution writer: output solutions, results, and evolution logs."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from src.core import TSPSolution, EvolutionTrajectory


class SolutionWriter:
    """Write solutions and results to disk."""

    @staticmethod
    def write_solution(path: str, solution: TSPSolution, metadata: Dict = None):
        """Write a single solution in readable format."""
        lines = [
            f"# TSP Solution",
            f"Method: {solution.method}",
            f"Distance: {solution.distance:.4f}",
            f"Runtime: {solution.runtime if solution.runtime else 'N/A'}",
            f"Tour: {' -> '.join(str(i) for i in solution.tour)}",
        ]
        if metadata:
            for k, v in metadata.items():
                lines.append(f"{k}: {v}")
        Path(path).write_text("\n".join(lines))

    @staticmethod
    def write_results(path: str, results: Dict[str, Dict[str, Any]]):
        """Write benchmark results as JSON."""
        Path(path).write_text(json.dumps(results, indent=2, ensure_ascii=False))

    @staticmethod
    def write_trajectory(path: str, trajectory: EvolutionTrajectory):
        """Write evolution trajectory as JSON."""
        data = []
        for c in trajectory.candidates:
            data.append({
                "candidate_id": c.candidate_id,
                "parents": c.parents,
                "generation": c.generation,
                "strategy": c.generation_strategy,
                "design_vector": c.design_vector,
                "description": c.description,
                "rationale": c.rationale,
                "evaluation": c.evaluation_results,
            })
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))

    @staticmethod
    def write_candidate_code(path: str, candidate_id: str, code: str):
        """Save generated algorithm code."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code)

    @staticmethod
    def write_report(path: str, trajectory: EvolutionTrajectory):
        """Generate and save a markdown evolution report."""
        lines = [
            f"# Evolution Report",
            f"**Generated**: {datetime.now().isoformat()}",
            f"**Total Candidates**: {len(trajectory.candidates)}",
            f"**Best Candidate**: {trajectory.best_candidate_id} "
            f"(distance: {trajectory.best_distance:.2f})",
            "",
            "## Candidates",
        ]
        for c in trajectory.candidates:
            gap_str = ""
            if c.evaluation_results:
                for inst, res in c.evaluation_results.items():
                    g = res.get("gap")
                    gap_str += f"{inst}={g:.2f}% " if g is not None else f"{inst}=FAIL "
            lines.append(f"- **{c.candidate_id}** "
                         f"(gen {c.generation}, {c.generation_strategy}): {gap_str}")
        Path(path).write_text("\n".join(lines))
