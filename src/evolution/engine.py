"""Evolution engine: the main loop of generate → evaluate → record."""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Any

from src.core import (
    AlgorithmDescription, DesignSpace, TSPInstance,
    EvolutionCandidate, EvolutionTrajectory,
)
from src.abstraction.design_space import DesignSpaceConstructor
from src.evaluation.runner import BenchmarkRunner, eval_code
from src.io.solution_writer import SolutionWriter
from src.utils.llm_client import LLMClient


class EvolutionEngine:
    """Core evolution loop over the algorithm design space."""

    def __init__(self, llm_client: LLMClient, data_dir: str = "."):
        self.llm = llm_client
        self.trajectory = EvolutionTrajectory()
        self.design_space: Optional[DesignSpace] = None
        self.algorithms: Dict[str, AlgorithmDescription] = {}
        self.benchmark = BenchmarkRunner(tsplib_dir=f"{data_dir}/data/tsplib")
        self.writer = SolutionWriter()
        self.results_dir = Path(data_dir) / "data/results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.generation = 0

    def initialize(self, algorithms: List[AlgorithmDescription]) -> DesignSpace:
        """Load known algorithms and build design space."""
        for a in algorithms:
            self.algorithms[a.algorithm_id] = a

        constructor = DesignSpaceConstructor()
        self.design_space = constructor.scan_dimensions(algorithms)
        print(f"[init] {len(algorithms)} algorithms → "
              f"{len(self.design_space.dimensions)} design dimensions")
        return self.design_space

    def generate_candidates(self, strategy: str = "hybrid",
                            count: int = 3) -> List[EvolutionCandidate]:
        """Generate candidate solutions."""
        if not self.algorithms:
            raise ValueError("No algorithms loaded. Call initialize() first.")

        algo_list = list(self.algorithms.values())
        candidates = []

        for _ in range(count):
            if strategy == "hybrid":
                p1, p2 = random.sample(algo_list, 2)
                c = self._hybridize(p1, p2)
            elif strategy == "mutate":
                p = random.choice(algo_list)
                c = self._mutate(p)
            elif strategy == "abstract":
                c = self._abstract(algo_list[:3])
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            c.generation = self.generation
            candidates.append(c)
            self.trajectory.add_candidate(c)

        return candidates

    def evaluate_candidate(self, candidate: EvolutionCandidate,
                           instance_names: Optional[List[str]] = None) -> Dict:
        """Generate code for a candidate and evaluate it."""
        parents_info = [
            self.algorithms[pid].to_dict()
            for pid in candidate.parents if pid in self.algorithms
        ]

        code = self.llm.generate_candidate(
            design_vector=candidate.design_vector,
            parents_info=parents_info,
            tsp_instance_info={
                "problem": "TSP",
                "instances": instance_names or ["eil51"],
                "input_format": "list of (x, y) coordinates",
                "output_format": "list of node indices (0..n-1)",
            }
        )

        # Save generated code
        code_path = self.results_dir / f"{candidate.candidate_id}.py"
        code_path.write_text(code)
        candidate.code_path = str(code_path)

        # Evaluate
        results = self.benchmark.run(code, instance_names)
        candidate.evaluation_results = results
        self.trajectory.record_evaluation(candidate.candidate_id, results)
        return results

    def step(self, instance_names: Optional[List[str]] = None,
             strategy: str = "hybrid") -> List[EvolutionCandidate]:
        """One complete evolution step."""
        candidates = self.generate_candidates(strategy=strategy, count=2)
        for c in candidates:
            self.evaluate_candidate(c, instance_names)
        self.generation += 1
        self._save_state()
        return candidates

    def get_best(self) -> Optional[EvolutionCandidate]:
        if self.trajectory.best_candidate_id:
            for c in self.trajectory.candidates:
                if c.candidate_id == self.trajectory.best_candidate_id:
                    return c
        return None

    def generate_report(self) -> str:
        """Markdown report of the evolution so far."""
        lines = [
            f"# Evolution Report",
            f"**Generation**: {self.generation}",
            f"**Total Candidates**: {len(self.trajectory.candidates)}",
            f"**Best**: {self.trajectory.best_candidate_id} "
            f"(dist={self.trajectory.best_distance:.2f})",
            "",
            "## Candidates",
        ]
        for c in self.trajectory.candidates:
            parts = []
            for inst, res in c.evaluation_results.items():
                g = res.get("gap")
                parts.append(f"{inst}={g:.2f}%" if g is not None else f"{inst}=X")
            lines.append(f"- {c.candidate_id}: {' | '.join(parts)}")
        return "\n".join(lines)

    # ---- Internal generation strategies ----

    def _hybridize(self, a1: AlgorithmDescription,
                   a2: AlgorithmDescription) -> EvolutionCandidate:
        """Combine design dimensions from two parent algorithms."""
        prompt = (
            f"Design a new TSP algorithm by hybridizing these two:\n\n"
            f"A ({a1.algorithm_id}): {a1.core_innovation[:200]}\n"
            f"  rep={a1.representation.value} search={a1.search_type.value} "
            f"init={a1.initialization} neighbor={a1.neighborhood}\n"
            f"B ({a2.algorithm_id}): {a2.core_innovation[:200]}\n"
            f"  rep={a2.representation.value} search={a2.search_type.value} "
            f"init={a2.initialization} neighbor={a2.neighborhood}\n\n"
            f"Pick the best dimension from each. "
            f"Return JSON: {{'design_vector':dict, 'description':str, 'rationale':str}}"
        )
        parsed = self.llm._parse_json(self.llm._call_llm(prompt))
        cid = f"evo-{self.generation}-{a1.algorithm_id}X{a2.algorithm_id}"
        return EvolutionCandidate(
            candidate_id=cid,
            parents=[a1.algorithm_id, a2.algorithm_id],
            design_vector=parsed.get("design_vector", {}),
            description=parsed.get("description", ""),
            rationale=parsed.get("rationale", ""),
            generation_strategy="crossover",
        )

    def _mutate(self, algo: AlgorithmDescription) -> EvolutionCandidate:
        """Mutate one design dimension."""
        prompt = (
            f"Suggest a mutation for this TSP algorithm:\n"
            f"{algo.algorithm_id}: {json.dumps(algo.to_dict(), indent=2)}\n\n"
            f"Change exactly ONE design dimension. "
            f"Return JSON: {{'design_vector':dict, 'description':str, 'rationale':str}}"
        )
        parsed = self.llm._parse_json(self.llm._call_llm(prompt))
        cid = f"evo-{self.generation}-mut-{algo.algorithm_id}"
        return EvolutionCandidate(
            candidate_id=cid,
            parents=[algo.algorithm_id],
            design_vector=parsed.get("design_vector", {}),
            description=parsed.get("description", ""),
            rationale=parsed.get("rationale", ""),
            generation_strategy="mutation",
        )

    def _abstract(self, algos: List[AlgorithmDescription]) -> EvolutionCandidate:
        """Synthesize a new approach from multiple algorithms."""
        summaries = "\n".join(
            f"{a.algorithm_id}: key={a.core_innovation[:150]}"
            for a in algos
        )
        prompt = (
            f"After studying these TSP algorithms:\n{summaries}\n"
            f"Design a NOVEL TSP algorithm that synthesizes their principles.\n"
            f"Return JSON: {{'design_vector':dict, 'description':str, 'rationale':str}}"
        )
        parsed = self.llm._parse_json(self.llm._call_llm(prompt))
        cid = f"evo-{self.generation}-abstract"
        return EvolutionCandidate(
            candidate_id=cid,
            parents=[a.algorithm_id for a in algos],
            design_vector=parsed.get("design_vector", {}),
            description=parsed.get("description", ""),
            rationale=parsed.get("rationale", ""),
            generation_strategy="abstraction",
        )

    def _save_state(self):
        state = {
            "generation": self.generation,
            "best_candidate": self.trajectory.best_candidate_id,
            "best_distance": self.trajectory.best_distance,
        }
        (self.results_dir / "evolution_state.json").write_text(
            json.dumps(state, indent=2))
        self.writer.write_trajectory(
            str(self.results_dir / "trajectory.json"), self.trajectory)
