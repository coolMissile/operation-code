"""Code reader: read source code from collected repositories."""

import json
import subprocess
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass


@dataclass
class RepoInfo:
    """Metadata about a collected implementation."""
    name: str
    url: str
    language: str
    algorithm_type: str
    main_file: str
    description: str = ""
    commit_hash: str = ""


KNOWN_TSP_REPOS = [
    {
        "name": "pomo",
        "url": "https://github.com/yd-kwon/POMO.git",
        "language": "Python",
        "algorithm_type": "learning_based",
        "description": "POMO: RL-based TSP solving with multiple rollouts",
        "main_file": "problems/tsp/pomo_tsp.py",
    },
    {
        "name": "lkh",
        "url": "https://github.com/xyc0123/LKH.git",
        "language": "C",
        "algorithm_type": "iterative_improvement",
        "description": "LKH: Lin-Kernighan heuristic for TSP",
        "main_file": "LKH.c",
    },
    {
        "name": "concorde",
        "url": "https://github.com/jilk/concorde.git",
        "language": "C",
        "algorithm_type": "exact",
        "description": "Concorde TSP solver (branch-and-cut)",
        "main_file": "concorde.h",
    },
    {
        "name": "or-tools-tsp",
        "url": "https://github.com/google/or-tools.git",
        "language": "C++/Python",
        "algorithm_type": "hybrid",
        "description": "Google OR-Tools CP-SAT for TSP",
        "main_file": "ortools/constraint_solver/samples/tsp.py",
    },
    {
        "name": "acotsp",
        "url": "https://github.com/xyc0123/ACOTSP.git",
        "language": "C",
        "algorithm_type": "population_based",
        "description": "Ant Colony Optimization for TSP",
        "main_file": "acotsp.c",
    },
]


class CodeReader:
    """Read source code from collected repositories."""

    def __init__(self, data_dir: str = "data/implementations"):
        self.data_dir = Path(data_dir)

    def list_repos(self) -> List[RepoInfo]:
        """List all collected repositories."""
        repos = []
        for item in self.data_dir.iterdir():
            if item.is_dir() and (item / ".repo_info.json").exists():
                info = json.loads((item / ".repo_info.json").read_text())
                repos.append(RepoInfo(**info))
        return repos

    def read_main_code(self, repo_name: str) -> Optional[str]:
        """Read the main algorithm file from a repo."""
        repo_dir = self.data_dir / repo_name
        info_path = repo_dir / ".repo_info.json"
        if not info_path.exists():
            return None

        info = json.loads(info_path.read_text())
        main_path = repo_dir / info.get("main_file", "")

        if main_path.exists() and main_path.is_file():
            return main_path.read_text(encoding="utf-8", errors="ignore")

        # Fallback: find the most relevant source file
        for ext in [".py", ".c", ".cpp", ".java", ".jl", ".rs"]:
            files = list(repo_dir.rglob(f"*{ext}"))
            if files:
                return files[0].read_text(encoding="utf-8", errors="ignore")
        return None

    def read_all_codes(self) -> Dict[str, str]:
        """Read main code from all repos."""
        return {
            r.name: code for r in self.list_repos()
            if (code := self.read_main_code(r.name)) is not None
        }

    def clone_repo(self, url: str, name: str, main_file: str = "",
                   force: bool = False) -> Path:
        """Clone a single repository."""
        target = self.data_dir / name
        if target.exists():
            if force:
                import shutil
                shutil.rmtree(target)
            else:
                print(f"[skip] {name} already exists at {target}")
                return target

        print(f"[clone] {name} from {url}")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", url, str(target)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to clone {name}: {result.stderr}")

        result = subprocess.run(
            ["git", "-C", str(target), "rev-parse", "HEAD"],
            capture_output=True, text=True
        )
        commit = result.stdout.strip() if result.returncode == 0 else ""

        (target / ".repo_info.json").write_text(json.dumps({
            "name": name, "url": url,
            "language": self._detect_language(target),
            "algorithm_type": "heuristic",
            "main_file": main_file,
            "description": "",
            "commit_hash": commit,
        }, indent=2))
        return target

    def clone_known_repos(self):
        """Clone all known TSP repos."""
        for r in KNOWN_TSP_REPOS:
            try:
                self.clone_repo(
                    url=r["url"], name=r["name"],
                    main_file=r["main_file"]
                )
            except Exception as e:
                print(f"[error] {r['name']}: {e}")

    @staticmethod
    def _detect_language(repo_dir: Path) -> str:
        """Detect primary language of a repo."""
        extensions = {".py": "Python", ".c": "C", ".cpp": "C++",
                      ".java": "Java", ".jl": "Julia", ".rs": "Rust"}
        counts = {lang: 0 for lang in extensions.values()}
        for f in repo_dir.rglob("*"):
            if f.suffix in extensions:
                counts[extensions[f.suffix]] += 1
        return max(counts, key=counts.get) if any(counts.values()) else "Unknown"
