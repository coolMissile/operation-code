"""
Extractor: use LLM to read a single algorithm's code and extract
structured description of its core logic, design dimensions, and key operators.
"""

import json
import re
from typing import Optional, Dict, Any

from src.core import AlgorithmDescription, SearchType, RepresentationType
from src.utils.llm_client import LLMClient


class AlgorithmExtractor:
    """Extract structured algorithm descriptions from source code using LLM."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def extract(self, algo_id: str, code: str,
                source: str = "", language: str = "Python") -> Optional[AlgorithmDescription]:
        """Extract algorithm description from source code.

        Tries LLM first, falls back to rule-based local extraction.
        """
        try:
            result = self.llm.extract_algorithm(code)
            if "raw" not in result:
                return self._build_description(algo_id, code, source, language, result)
        except Exception as e:
            print(f"  [LLM] Extraction failed: {e}")

        return self._extract_local(algo_id, code, source, language)

    def _build_description(self, algo_id: str, code: str,
                           source: str, language: str,
                           result: Dict[str, Any]) -> AlgorithmDescription:
        """Build AlgorithmDescription from LLM JSON result."""
        def safe_enum(enum_cls, key):
            val = result.get(key, "")
            try:
                return enum_cls(val)
            except (ValueError, KeyError):
                return list(enum_cls)[0]

        return AlgorithmDescription(
            algorithm_id=algo_id,
            name=algo_id,
            source=source,
            language=language,
            representation=safe_enum(RepresentationType, "representation"),
            search_type=safe_enum(SearchType, "search_type"),
            initialization=result.get("initialization", "unknown"),
            neighborhood=result.get("neighborhood", "unknown"),
            selection=result.get("selection", "unknown"),
            acceptance=result.get("acceptance", "unknown"),
            population=result.get("population_strategy", "unknown"),
            termination=result.get("termination", "unknown"),
            core_innovation=result.get("core_innovation", ""),
            pseudocode=result.get("pseudocode", ""),
            key_operators=result.get("key_operators", []),
            dependencies=[],
            tsp_format="tsplib",
            raw_code_path=source,
        )

    def _extract_local(self, algo_id: str, code: str,
                       source: str, language: str) -> AlgorithmDescription:
        """Rule-based extraction without LLM."""
        text = code.lower()

        # --- Search type ---
        search_type = SearchType.ITERATIVE_IMPROVEMENT
        if any(kw in text for kw in ["population", "generation", "crossover", "genetic"]):
            search_type = SearchType.POPULATION_BASED
        elif any(kw in text for kw in ["reinforce", "policy", "attention", "encoder", "decoder"]):
            search_type = SearchType.LEARNING_BASED
        elif any(kw in text for kw in ["greedy", "nearest", "insert"]):
            search_type = SearchType.CONSTRUCTIVE
        elif any(kw in text for kw in ["branch", "cut", "integer", "linear_program"]):
            search_type = SearchType.EXACT

        # --- Representation ---
        representation = RepresentationType.PERMUTATION
        if any(kw in text for kw in ["heatmap", "attention", "logit"]):
            representation = RepresentationType.HEATMAP
        elif "adjacency" in text:
            representation = RepresentationType.ADJACENCY
        elif "probability" in text:
            representation = RepresentationType.PROBABILITY_MATRIX

        # --- Operators ---
        operators = []
        patterns = [
            (r"2-?opt", "2-opt"),
            (r"\bswap\b", "swap"),
            (r"insert", "insertion"),
            (r"crossover|pmx|order_x", "crossover"),
            (r"mutate|mutation", "mutation"),
            (r"tournament", "tournament_selection"),
            (r"roulette", "roulette_selection"),
            (r"elite", "elitism"),
            (r"lk(|_search)", "LK_search"),
            (r"local.search", "local_search"),
            (r"nearest.neighbor", "nearest_neighbor"),
        ]
        for pattern, op_name in patterns:
            if re.search(pattern, text):
                operators.append(op_name)

        return AlgorithmDescription(
            algorithm_id=algo_id,
            name=algo_id,
            source=source,
            language=language,
            representation=representation,
            search_type=search_type,
            initialization=self._grep(text, "nearest|greedy|random|christofides", "random"),
            neighborhood=", ".join(operators) if operators else "local_search",
            selection=self._grep(text, "tournament|roulette|rank|random", "deterministic"),
            acceptance=self._grep(text, "improving|annealing|boltzmann|greedy", "improving"),
            population=self._grep(text, "steady|generational|single|multiple", "single"),
            termination=self._grep(text, "iteration|generation|convergence|time_limit", "iterations"),
            core_innovation=f"{search_type.value} solver using {', '.join(operators) if operators else 'standard'} operators.",
            pseudocode=f"1. Initialize solution\n2. Apply {', '.join(operators) if operators else 'search'}\n3. Return best",
            key_operators=operators,
            dependencies=[language],
            tsp_format="tsplib" if "tsplib" in text else "custom",
            raw_code_path=source,
        )

    @staticmethod
    def _grep(text: str, patterns: str, default: str) -> str:
        m = re.search(patterns, text)
        return m.group(0) if m else default
