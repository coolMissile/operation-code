#!/usr/bin/env python3
"""
CodeEvolution-CO: Code Evolution for Combinatorial Optimization.

Pipeline:
  collect       → Clone open-source TSP repos
  analyze       → Extract algorithm descriptions from code
  design-space  → Build the design space from extracted algorithms
  evolve        → Run the evolution loop
  all           → Full pipeline

Usage:
  python run_pipeline.py collect
  python run_pipeline.py analyze [--mock]
  python run_pipeline.py design-space
  python run_pipeline.py evolve [--generations N]
  python run_pipeline.py all [--mock]
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.io.code_reader import CodeReader
from src.io.problem_reader import ProblemReader
from src.io.solution_writer import SolutionWriter
from src.abstraction.extractor import AlgorithmExtractor
from src.abstraction.abstractor import AlgorithmAbstractor
from src.abstraction.design_space import DesignSpaceConstructor
from src.evolution.engine import EvolutionEngine
from src.evaluation.runner import BenchmarkRunner
from src.utils.llm_client import LLMClient


# ────────────────────────── Commands ──────────────────────────

def cmd_collect(args):
    """Clone open-source TSP solver repos."""
    reader = CodeReader("data/implementations")
    print("[collect] Cloning known TSP repos...")
    reader.clone_known_repos()
    repos = reader.list_repos()
    print(f"[collect] Collected {len(repos)} implementations:")
    for r in repos:
        print(f"  - {r.name}: {r.description or '(no description)'}")


def cmd_analyze(args):
    """Extract algorithm descriptions from collected code."""
    reader = CodeReader("data/implementations")
    repos = reader.list_repos()

    if not repos:
        print("[analyze] No repos found. Run 'collect' first.")
        return

    llm = LLMClient()
    extractor = AlgorithmExtractor(llm)
    algorithms = []

    for repo in repos:
        print(f"[analyze] Extracting {repo.name}...")
        code = reader.read_main_code(repo.name)
        if not code:
            print(f"  [skip] No source code for {repo.name}")
            continue
        algo = extractor.extract(
            algo_id=repo.name, code=code,
            source=repo.url, language=repo.language
        )
        if algo:
            algorithms.append(algo)
            print(f"  → {algo.search_type.value}, ops={algo.key_operators[:3]}")

    if not algorithms:
        print("[analyze] No algorithms extracted.")
        return

    # Save
    out_path = Path("data/results/algorithms.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps([a.to_dict() for a in algorithms], indent=2, ensure_ascii=False)
    )
    print(f"[analyze] Saved {len(algorithms)} algorithms to {out_path}")

    # Also produce a human-readable summary
    summary_path = Path("data/results/algorithms_summary.md")
    lines = ["# Extracted Algorithms\n"]
    for a in algorithms:
        lines.append(f"## {a.algorithm_id}")
        lines.append(f"- Search: {a.search_type.value}")
        lines.append(f"- Rep: {a.representation.value}")
        lines.append(f"- Init: {a.initialization}")
        lines.append(f"- Neighbor: {a.neighborhood}")
        lines.append(f"- Innovation: {a.core_innovation[:150]}")
        lines.append("")
    summary_path.write_text("\n".join(lines))
    print(f"[analyze] Summary saved to {summary_path}")


def cmd_design_space(args):
    """Build the design space from extracted algorithms."""
    algo_path = Path("data/results/algorithms.json")
    if not algo_path.exists():
        print("[design-space] Run 'analyze' first.")
        return

    algorithms = _load_algorithms(algo_path)
    constructor = DesignSpaceConstructor()
    space = constructor.scan_dimensions(algorithms)

    # Markdown report
    report = constructor.design_space_summary(space)
    report_path = Path("data/results/design_space.md")
    report_path.write_text(report)
    print(report)


def cmd_evolve(args):
    """Run the evolution loop."""
    algo_path = Path("data/results/algorithms.json")
    if not algo_path.exists():
        print("[evolve] Run 'analyze' first.")
        return

    algorithms = _load_algorithms(algo_path)
    llm = LLMClient()
    engine = EvolutionEngine(llm, data_dir=".")

    engine.initialize(algorithms)
    print(f"[evolve] Initialized with {len(algorithms)} algorithms")

    instances = ["eil51", "berlin52"]
    for gen in range(args.generations):
        print(f"\n[evolve] === Generation {gen + 1} ===")
        candidates = engine.step(instance_names=instances, strategy="hybrid")
        for c in candidates:
            parts = []
            for inst, res in c.evaluation_results.items():
                g = res.get("gap")
                parts.append(f"{inst}: dist={res.get('distance',0):.1f}"
                             f"{', gap='+f'{g:.2f}%' if g is not None else ''}")
            print(f"  {c.candidate_id} ({c.generation_strategy}): {' | '.join(parts)}")

    print("\n" + engine.generate_report())

    # Save report
    engine.writer.write_report(
        str(Path("data/results/evolution_report.md")),
        engine.trajectory
    )


def cmd_all(args):
    """Full pipeline."""
    cmd_collect(args)
    cmd_analyze(args)
    cmd_design_space(args)
    cmd_evolve(args)


# ────────────────────────── Helpers ──────────────────────────

def _load_algorithms(path):
    """Load AlgorithmDescription list from JSON."""
    from src.core import AlgorithmDescription, SearchType, RepresentationType

    with open(path) as f:
        items = json.load(f)

    algorithms = []
    for d in items:
        try:
            st = SearchType(d.get("search_type", "iterative_improvement"))
        except ValueError:
            st = SearchType.ITERATIVE_IMPROVEMENT
        try:
            rt = RepresentationType(d.get("representation", "permutation"))
        except ValueError:
            rt = RepresentationType.PERMUTATION

        algorithms.append(AlgorithmDescription(
            algorithm_id=d["algorithm_id"],
            name=d.get("name", d["algorithm_id"]),
            source=d.get("source", ""),
            language=d.get("language", "Python"),
            representation=rt,
            search_type=st,
            initialization=d.get("initialization", ""),
            neighborhood=d.get("neighborhood", ""),
            selection=d.get("selection", ""),
            acceptance=d.get("acceptance", ""),
            population=d.get("population_strategy", ""),
            termination=d.get("termination", ""),
            core_innovation=d.get("core_innovation", ""),
            pseudocode=d.get("pseudocode", ""),
            key_operators=d.get("key_operators", []),
        ))
    return algorithms


# ────────────────────────── Main ──────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CodeEvolution-CO: Code Evolution for Combinatorial Optimization"
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("collect", help="Clone TSP solver repos")
    sub.add_parser("analyze", help="Extract algorithms from code")
    sub.add_parser("design-space", help="Build design space")

    p = sub.add_parser("evolve", help="Run evolution loop")
    p.add_argument("--generations", type=int, default=3)

    sub.add_parser("all", help="Run full pipeline")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    cmds = {
        "collect": cmd_collect,
        "analyze": cmd_analyze,
        "design-space": cmd_design_space,
        "evolve": cmd_evolve,
        "all": cmd_all,
    }
    cmds[args.command](args)


if __name__ == "__main__":
    main()
