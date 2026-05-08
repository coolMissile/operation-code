"""Problem instance reader: parse TSPLIB and other formats into TSPInstance."""

from pathlib import Path
from typing import List, Optional, Tuple, Dict

from src.core import TSPInstance


# Known optimal values for TSPLIB instances
TSPLIB_OPTIMA: Dict[str, float] = {
    "eil51": 426.0, "berlin52": 7542.0, "st70": 675.0, "eil76": 538.0,
    "pr76": 108159.0, "rat99": 1211.0, "kroA100": 21282.0, "kroB100": 22141.0,
    "ch150": 6528.0, "tsp225": 3919.0, "a280": 2579.0, "lin318": 42029.0,
    "pcb442": 50778.0, "d493": 35002.0, "rat575": 6773.0,
}


class ProblemReader:
    """Read problem instances from various formats."""

    @staticmethod
    def from_tsplib(filepath: str) -> TSPInstance:
        """Parse a TSPLIB-format file into a TSPInstance."""
        name = ""
        nodes: List[Tuple[float, float]] = []
        section = "header"
        comment_lines = []

        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("NAME:"):
                    name = line.split(":", 1)[1].strip()
                elif line.startswith("NAME "):
                    name = line.split(None, 1)[1].strip()
                elif line.startswith("COMMENT"):
                    comment_lines.append(line.split(":", 1)[1].strip())
                elif line.startswith("DIMENSION"):
                    pass
                elif line == "NODE_COORD_SECTION":
                    section = "coords"
                elif line == "EOF":
                    break
                elif section == "coords":
                    parts = line.split()
                    if len(parts) >= 3:
                        nodes.append((float(parts[1]), float(parts[2])))

        optimal = TSPLIB_OPTIMA.get(name)
        return TSPInstance(
            name=name or Path(filepath).stem,
            nodes=nodes,
            optimal=float(optimal) if optimal else None,
            comment=" | ".join(comment_lines),
        )

    @staticmethod
    def from_coordinates(name: str, coords: List[Tuple[float, float]],
                         optimal: Optional[float] = None) -> TSPInstance:
        """Create instance directly from coordinate list."""
        return TSPInstance(name=name, nodes=coords, optimal=optimal)

    @staticmethod
    def random_instance(name: str, num_nodes: int, seed: int = 42,
                        optimal: Optional[float] = None) -> TSPInstance:
        """Generate a random TSP instance."""
        import random
        random.seed(seed)
        nodes = [(random.uniform(0, 100), random.uniform(0, 100))
                 for _ in range(num_nodes)]
        return TSPInstance(name=name, nodes=nodes, optimal=optimal)

    @staticmethod
    def load_tsplib_dir(tsplib_dir: str,
                        instance_names: Optional[List[str]] = None
                        ) -> Dict[str, TSPInstance]:
        """Load multiple TSPLIB instances from a directory."""
        tsplib_path = Path(tsplib_dir)
        instances = {}

        if instance_names:
            names = instance_names
        else:
            names = [f.stem for f in tsplib_path.glob("*.tsp")]

        for name in names:
            for ext in [".tsp", ".txt", ""]:
                path = tsplib_path / f"{name}{ext}"
                if path.exists():
                    instances[name] = ProblemReader.from_tsplib(str(path))
                    break
        return instances
