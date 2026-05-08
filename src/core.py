"""Core abstractions for the CodeEvolution-CO framework."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum


# ========== Problem Layer ==========

@dataclass
class TSPInstance:
    """Unified TSP problem instance."""
    name: str
    nodes: List[Tuple[float, float]]  # (x, y) coordinates
    optimal: Optional[float] = None
    comment: str = ""

    @property
    def dimension(self) -> int:
        return len(self.nodes)

    def distance_matrix(self) -> List[List[float]]:
        """Compute Euclidean distance matrix."""
        n = self.dimension
        mat = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = ((self.nodes[i][0] - self.nodes[j][0]) ** 2 +
                     (self.nodes[i][1] - self.nodes[j][1]) ** 2) ** 0.5
                mat[i][j] = mat[j][i] = d
        return mat


@dataclass
class TSPSolution:
    """A solution to a TSP instance."""
    tour: List[int]  # permutation of node indices
    distance: float
    method: str = "unknown"
    runtime: Optional[float] = None

    def validate(self, instance: TSPInstance) -> bool:
        """Check if tour is a valid Hamiltonian cycle."""
        n = instance.dimension
        if len(self.tour) != n:
            return False
        if sorted(self.tour) != list(range(n)):
            return False
        return True

    def gap(self, instance: TSPInstance) -> Optional[float]:
        """Gap to optimal (percentage)."""
        if instance.optimal is None:
            return None
        return (self.distance - instance.optimal) / instance.optimal * 100


# ========== Algorithm Knowledge Layer ==========

class SearchType(Enum):
    CONSTRUCTIVE = "constructive"
    ITERATIVE_IMPROVEMENT = "iterative_improvement"
    POPULATION_BASED = "population_based"
    LEARNING_BASED = "learning_based"
    EXACT = "exact"
    HYBRID = "hybrid"


class RepresentationType(Enum):
    PERMUTATION = "permutation"
    HEATMAP = "heatmap"
    EDGE_LIST = "edge_list"
    ADJACENCY = "adjacency"
    PROBABILITY_MATRIX = "probability_matrix"


@dataclass
class AlgorithmDescription:
    """Structured description of an algorithm extracted from code."""
    algorithm_id: str
    name: str
    source: str  # repo URL or file path
    language: str  # Python, C++, Julia, etc.

    # Design dimensions
    representation: RepresentationType
    search_type: SearchType
    initialization: str
    neighborhood: str
    selection: str
    acceptance: str
    population: str
    termination: str

    # Core logic (LLM extracted)
    core_innovation: str = ""
    pseudocode: str = ""
    key_operators: List[str] = field(default_factory=list)

    # Metadata
    dependencies: List[str] = field(default_factory=list)
    tsp_format: str = ""  # TSPlib, coordinates, distance_matrix
    raw_code_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm_id": self.algorithm_id,
            "name": self.name,
            "source": self.source,
            "language": self.language,
            "representation": self.representation.value,
            "search_type": self.search_type.value,
            "initialization": self.initialization,
            "neighborhood": self.neighborhood,
            "selection": self.selection,
            "acceptance": self.acceptance,
            "population": self.population,
            "termination": self.termination,
            "core_innovation": self.core_innovation,
            "pseudocode": self.pseudocode,
            "key_operators": self.key_operators,
            "dependencies": self.dependencies,
            "tsp_format": self.tsp_format,
        }


# ========== Design Space Layer ==========

@dataclass
class DesignDimension:
    """A dimension in the design space with possible values."""
    name: str
    possible_values: List[str]
    description: str = ""

    def __hash__(self):
        return hash(self.name)


@dataclass
class DesignVector:
    """A point in the design space (a specific algorithm configuration)."""
    dimensions: Dict[str, str]  # dimension_name -> value

    def __hash__(self):
        return hash(tuple(sorted(self.dimensions.items())))


@dataclass
class DesignSpace:
    """The complete design space with known and unknown points."""
    dimensions: List[DesignDimension]
    known_points: Dict[str, DesignVector]  # algorithm_id -> vector
    name: str = "TSP Design Space"

    def unknown_combinations(self, top_k: int = 10) -> List[Dict[str, str]]:
        """Suggest unexplored design combinations."""
        import itertools
        all_values = [d.possible_values for d in self.dimensions]
        dim_names = [d.name for d in self.dimensions]
        known_vectors = set(
            tuple(sorted(v.dimensions.items()))
            for v in self.known_points.values()
        )

        candidates = []
        for combo in itertools.product(*all_values):
            vec = tuple(sorted(zip(dim_names, combo)))
            if vec not in known_vectors:
                candidates.append(dict(vec))
                if len(candidates) >= top_k:
                    break
        return candidates


# ========== Evolution Layer ==========

@dataclass
class EvolutionCandidate:
    """A candidate solution generated by the evolution engine."""
    candidate_id: str
    parents: List[str]  # algorithm_ids
    design_vector: Dict[str, str]
    description: str
    rationale: str
    generation_strategy: str  # crossover / mutation / abstraction

    # Evaluation results
    evaluation_results: Dict[str, Any] = field(default_factory=dict)
    code_path: Optional[str] = None
    generation: int = 0


@dataclass
class EvolutionTrajectory:
    """Record of the evolution process."""
    candidates: List[EvolutionCandidate] = field(default_factory=list)
    best_distance: Optional[float] = None
    best_candidate_id: Optional[str] = None
    generation_count: int = 0

    def add_candidate(self, candidate: EvolutionCandidate):
        self.candidates.append(candidate)

    def record_evaluation(self, candidate_id: str, results: Dict[str, Any]):
        for c in self.candidates:
            if c.candidate_id == candidate_id:
                c.evaluation_results = results
                dist = results.get("distance", float("inf"))
                if self.best_distance is None or dist < self.best_distance:
                    self.best_distance = dist
                    self.best_candidate_id = candidate_id
                break
