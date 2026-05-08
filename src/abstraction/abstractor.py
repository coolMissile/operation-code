"""
Abstractor: analyze multiple extracted algorithm descriptions to find
commonalities, differences, and synthesize new algorithmic ideas.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.core import AlgorithmDescription, DesignDimension, DesignVector, DesignSpace
from src.abstraction.design_space import DesignSpaceConstructor
from src.utils.llm_client import LLMClient


@dataclass
class CrossAlgorithmInsight:
    """An insight derived from comparing multiple algorithms."""
    insight_type: str  # "common_pattern" | "unique_feature" | "gap" | "synthesis"
    description: str
    involved_algorithms: List[str]
    significance: str  # "high" | "medium" | "low"


class AlgorithmAbstractor:
    """Analyze multiple algorithms to find patterns and synthesize new ideas."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def find_common_patterns(self, algorithms: List[AlgorithmDescription]) -> List[CrossAlgorithmInsight]:
        """Identify patterns shared across multiple algorithms."""
        insights = []

        # --- Structural commonalities ---
        # Group by design dimensions
        from collections import Counter
        for dim_name in ["representation", "search_type", "acceptance"]:
            values = Counter()
            for a in algorithms:
                v = getattr(a, dim_name, None)
                if v:
                    values[v.value if hasattr(v, 'value') else str(v)] += 1

            for val, count in values.most_common(2):
                if count >= len(algorithms) * 0.5:
                    insights.append(CrossAlgorithmInsight(
                        insight_type="common_pattern",
                        description=f"Most algorithms use {val} for {dim_name} ({count}/{len(algorithms)})",
                        involved_algorithms=[a.algorithm_id for a in algorithms
                                             if getattr(a, dim_name, None) and
                                             (getattr(a, dim_name).value if hasattr(getattr(a, dim_name), 'value') else str(getattr(a, dim_name))) == val],
                        significance="high" if count == len(algorithms) else "medium",
                    ))

        # --- Unique features ---
        for a in algorithms:
            unique_aspects = []
            if a.core_innovation and "standard" not in a.core_innovation.lower():
                unique_aspects.append(a.core_innovation[:100])
            if unique_aspects:
                insights.append(CrossAlgorithmInsight(
                    insight_type="unique_feature",
                    description=f"{a.algorithm_id}: {'; '.join(unique_aspects)}",
                    involved_algorithms=[a.algorithm_id],
                    significance="medium",
                ))

        # --- Gaps: identify design dimensions not covered ---
        all_ops = set()
        for a in algorithms:
            all_ops.update(a.key_operators)

        # Check if 2-opt is present (core TSP operator)
        if "2-opt" not in all_ops:
            insights.append(CrossAlgorithmInsight(
                insight_type="gap",
                description="2-opt operator not found in collected algorithms",
                involved_algorithms=[],
                significance="high",
            ))

        return insights

    def synthesize_new_approach(self, algorithms: List[AlgorithmDescription],
                                 insight: CrossAlgorithmInsight) -> str:
        """Use LLM to design a new algorithm addressing a gap or combining patterns."""
        algo_summaries = "\n\n".join(
            f"Algorithm {a.algorithm_id}:\n"
            f"  Design: representation={a.representation.value}, "
            f"search={a.search_type.value}, "
            f"neighborhood={a.neighborhood}, "
            f"selection={a.selection}\n"
            f"  Innovation: {a.core_innovation[:200]}"
            for a in algorithms
        )

        prompt = f"""You are studying a collection of TSP algorithms:

{aligned_summaries}

Key insight: {insight.description}

Based on this analysis, design a NEW algorithmic approach that addresses
the insight above. Describe:
1. The core idea
2. How it differs from existing approaches
3. The key algorithmic steps
4. Why it could be effective

Return your design as structured text."""
        return self.llm._call_llm(prompt)

    def rank_algorithm_family(self, algorithms: List[AlgorithmDescription]) -> List[Dict]:
        """Group algorithms into families and rank by diversity."""
        families: Dict[str, List[AlgorithmDescription]] = {}
        for a in algorithms:
            key = a.search_type.value
            if key not in families:
                families[key] = []
            families[key].append(a)

        ranked = []
        for family_type, members in families.items():
            ranked.append({
                "family": family_type,
                "count": len(members),
                "members": [m.algorithm_id for m in members],
                "representations": list(set(m.representation.value for m in members)),
                "operators": list(set(op for m in members for op in m.key_operators)),
            })

        ranked.sort(key=lambda x: x["count"], reverse=True)
        return ranked
