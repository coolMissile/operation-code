"""Configuration for the CodeEvolution-CO framework."""

import os
from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMPLEMENTATIONS_DIR = DATA_DIR / "implementations"
TSPLIB_DIR = DATA_DIR / "tsplib"
RESULTS_DIR = DATA_DIR / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# LLM configuration
LLM_PROVIDER = os.environ.get("CO_EVO_LLM_PROVIDER", "deepseek")
LLM_MODEL = os.environ.get("CO_EVO_LLM_MODEL", "deepseek-chat")
LLM_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# Evolution
DEFAULT_STRATEGIES = ["hybrid", "mutate", "abstract"]
POPULATION_SIZE = 10
MAX_GENERATIONS = 20

# Evaluation
BENCHMARK_INSTANCES = ["eil51", "berlin52", "kroA100"]
TIMEOUT_SECONDS = 300

# Ensure directories exist
for d in [DATA_DIR, IMPLEMENTATIONS_DIR, TSPLIB_DIR, RESULTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
