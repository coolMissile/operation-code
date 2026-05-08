#!/usr/bin/env python3
"""Verify the refactored architecture works end-to-end (no LLM API needed)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.core import (
    TSPInstance, TSPSolution, AlgorithmDescription,
    SearchType, RepresentationType, DesignSpace,
    EvolutionCandidate, EvolutionTrajectory,
)
from src.io.problem_reader import ProblemReader
from src.io.solution_writer import SolutionWriter
from src.io.code_reader import CodeReader
from src.abstraction.extractor import AlgorithmExtractor
from src.abstraction.abstractor import AlgorithmAbstractor
from src.abstraction.design_space import DesignSpaceConstructor
from src.evolution.engine import EvolutionEngine
from src.evaluation.runner import BenchmarkRunner, eval_code
from src.utils.llm_client import LLMClient

print("=" * 60)
print("CodeEvolution-CO: Architecture Verification")
print("=" * 60)


# ── 1. Core data structures ──
print("\n[1/6] Core data structures...")
inst = TSPInstance(name="t5", nodes=[(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)], optimal=4.0)
assert inst.dimension == 5
dm = inst.distance_matrix()
assert abs(dm[0][1] - 1.0) < 0.001
sol = TSPSolution(tour=[0, 1, 2, 3, 4], distance=4.0, method="test")
assert sol.validate(inst)
assert abs(sol.gap(inst) - 0.0) < 0.001
print("  TSPInstance + TSPSolution: OK")

# ── 2. IO: ProblemReader ──
print("\n[2/6] IO module...")
reader = ProblemReader()
rand_inst = reader.random_instance("test_rand", 10, seed=42)
assert rand_inst.dimension == 10
print(f"  ProblemReader.random_instance: OK ({rand_inst.dimension} nodes)")

# CodeReader (structurally)
cr = CodeReader("data/implementations")
print(f"  CodeReader: OK (looks in {cr.data_dir})")

# SolutionWriter
writer = SolutionWriter()
print("  SolutionWriter: OK")

# ── 3. Abstraction: Extractor (local mode) ──
print("\n[3/6] Abstraction: local extractor...")
mock_code = """
class GeneticTSP:
    def __init__(self, pop_size=100):
        self.pop = [list(range(n)) for _ in range(pop_size)]
    def evolve(self):
        while not converged:
            parents = tournament_select(self.pop)
            child = crossover(parents[0], parents[1])
            if random() < 0.1:
                mutate(child)
            self.pop = generational_replace(self.pop, child)
"""
extractor = AlgorithmExtractor(LLMClient())
algo = extractor._extract_local("genetic", mock_code, "mock", "Python")
assert algo.search_type == SearchType.POPULATION_BASED
assert algo.representation == RepresentationType.PERMUTATION
assert "crossover" in algo.key_operators
assert "mutation" in algo.key_operators
print(f"  {algo.algorithm_id}: {algo.search_type.value} / ops={algo.key_operators}")

# ── 4. Abstraction: Abstractor + DesignSpace ──
print("\n[4/6] Abstraction: abstractor + design space...")

# Build a few mock algorithms
algos = [
    extractor._extract_local("ga", "population crossover mutation", "mock", "Python"),
    extractor._extract_local("lkh", "2-opt local search", "mock", "Python"),
    extractor._extract_local("pomo", "attention encoder decoder reinforce", "mock", "Python"),
]

# Abstractor
abstractor = AlgorithmAbstractor(LLMClient())
insights = abstractor.find_common_patterns(algos)
print(f"  Insights found: {len(insights)}")
for ins in insights:
    print(f"    [{ins.insight_type}] {ins.description[:80]}")

families = abstractor.rank_algorithm_family(algos)
print(f"  Algorithm families: {len(families)}")
for f in families:
    print(f"    {f['family']}: {f['members']}")

# DesignSpace
constructor = DesignSpaceConstructor()
space = constructor.scan_dimensions(algos)
assert len(space.dimensions) > 0
unexplored = constructor.suggest_unexplored(space, top_k=3)
print(f"  Design space: {len(space.dimensions)} dims, {len(unexplored)} suggested unexplored")

# ── 5. Evaluation ──
print("\n[5/6] Evaluation runner...")
runner = BenchmarkRunner()
results = runner.run_on_random("", sizes=[10])  # empty code → will fail
assert "random_10" in results
print(f"  BenchmarkRunner: OK (random_10 reported)")

# Test eval_code with a valid solver snippet
valid_code = """
import math
def solve():
    n = dimension
    tour = list(range(n))
    dist = sum(math.dist(nodes[tour[i]], nodes[tour[(i+1)%n]]) for i in range(n))
    return tour
"""
inst10 = reader.random_instance("vtest", 10, seed=42)
res = eval_code(valid_code, inst10)
assert res["success"] is True
print(f"  eval_code (valid solver): distance={res['distance']:.2f}")

# ── 6. Evolution Engine (structural) ──
print("\n[6/6] Evolution engine...")
llm = LLMClient()
engine = EvolutionEngine(llm, data_dir=".")
space = engine.initialize(algos)
assert len(engine.algorithms) == 3
print(f"  Engine initialized: {len(engine.algorithms)} algorithms")

# Test trajectory
c = EvolutionCandidate(
    candidate_id="evo-test-1",
    parents=["ga", "lkh"],
    design_vector={"representation": "permutation", "search": "hybrid"},
    description="Hybrid test",
    rationale="Test",
    generation_strategy="crossover",
)
engine.trajectory.add_candidate(c)
engine.trajectory.record_evaluation(c.candidate_id, {"distance": 100.0, "gap": 5.0})
assert engine.trajectory.best_candidate_id == "evo-test-1"
assert engine.trajectory.best_distance == 100.0
print("  Evolution trajectory recording: OK")


print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
print("""
Module structure:
  src/
    core.py                       Core data structures
    io/
      problem_reader.py           Read problem instances
      solution_writer.py          Write solutions & reports
      code_reader.py              Read source code from repos
    abstraction/
      extractor.py                LLM extraction from single code
      abstractor.py               Cross-algorithm analysis + synthesis
      design_space.py             Design space construction
    evolution/
      engine.py                   Evolution loop
    evaluation/
      runner.py                   Benchmark evaluation
    utils/
      llm_client.py               LLM API client

Commands:
  python run_pipeline.py collect
  python run_pipeline.py analyze
  python run_pipeline.py design-space
  python run_pipeline.py evolve --generations 3
  python run_pipeline.py all
""")
