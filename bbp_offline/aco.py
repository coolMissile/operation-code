#1D bbp offline

import numpy as np
import numpy.typing as npt
from typing import List, Tuple
from math import floor

IntArray = npt.NDArray[np.int_]
FloatArray = npt.NDArray[np.float_]


# -----------------------------
# Utility
# -----------------------------
def organize_path(path: IntArray) -> Tuple[int, IntArray]:
    order = {}
    result = np.zeros_like(path)
    for i, v in enumerate(path):
        if v in order:
            result[i] = order[v]
        else:
            result[i] = order[v] = len(order)
    return len(order), result


def path_fitness(vacancies: List[int], capacity: int) -> float:
    occupied = capacity - np.array(vacancies, dtype=float)
    return ((occupied / capacity) ** 2).sum().item() / len(vacancies)


class UniformGenerator:
    def __init__(self, batch_size=500):
        self.batch_size = batch_size
        self.numbers = np.random.random(batch_size)
        self.idx = 0

    def next(self) -> float:
        if self.idx >= self.batch_size:
            self.numbers = np.random.random(self.batch_size)
            self.idx = 0
        val = self.numbers[self.idx]
        self.idx += 1
        return val


# -----------------------------
# ACO Solver for BPP
# -----------------------------
class ACO_BPP_Solver:
    def __init__(
        self,
        demand: IntArray,
        capacity: int,
        n_ants: int = 30,
        iterations: int = 100,
        decay: float = 0.95,
        alpha: float = 1.0,
        beta: float = 2.0,
        seed: int | None = None,
    ):
        if seed is not None:
            np.random.seed(seed)

        self.problem_size = len(demand)
        self.capacity = capacity
        self.demand = demand.copy()
        assert self.demand.max() <= self.capacity

        self.n_ants = n_ants
        self.iterations = iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

        # pheromone & heuristic
        self.pheromone = np.ones((self.problem_size, self.problem_size))
        self.heuristic = self._build_heuristic()

        self.uniform_gen = UniformGenerator()

        self.best_path = np.arange(self.problem_size)
        self.best_bins = self.problem_size

    # -----------------------------
    # Core API
    # -----------------------------
    def solve(self) -> Tuple[int, IntArray]:
        for _ in range(self.iterations):
            prob = self.pheromone ** self.alpha * self.heuristic ** self.beta
            paths, costs, fitnesses = self._gen_population(self.n_ants, prob)

            best_idx = costs.argmin()
            if costs[best_idx] < self.best_bins:
                self.best_bins = costs[best_idx]
                self.best_path = paths[best_idx]

            self._update_pheromone(paths, fitnesses)

        return organize_path(self.best_path)

    # -----------------------------
    # Internal methods
    # -----------------------------
    def _build_heuristic(self) -> FloatArray:
        """
        Heuristic: smaller items are preferred
        """
        demand = self.demand.astype(float)
        heuristic = demand / demand.max()
        heuristic = 1.0 - heuristic
        heuristic[heuristic < 1e-6] = 1e-6
        return np.tile(heuristic, (self.problem_size, 1))

    def _gen_population(
        self, count: int, prob: FloatArray
    ) -> Tuple[List[IntArray], IntArray, FloatArray]:
        paths, costs, fitnesses = [], [], []
        for _ in range(count):
            path, cost, fitness = self._sample_path(prob)
            paths.append(path)
            costs.append(cost)
            fitnesses.append(fitness)
        return paths, np.array(costs), np.array(fitnesses)

    def _sample_path(
        self, prob: FloatArray
    ) -> Tuple[IntArray, int, float]:
        path = np.full(self.problem_size, -1)
        valid = np.ones(self.problem_size, dtype=bool)

        bin_id = 0
        vacancy = self.capacity
        bin_items = np.zeros(self.problem_size, dtype=bool)
        item_count = 0

        vacancies = []

        for _ in range(self.problem_size):
            mask = (self.demand <= vacancy) & valid
            if not np.any(mask):
                vacancies.append(vacancy)
                bin_id += 1
                vacancy = self.capacity
                bin_items[:] = False
                item_count = 0
                mask = valid

            if item_count == 0:
                selected = self._random_select(mask)
            else:
                item_prob = (prob[bin_items].sum(0) / item_count) * mask
                selected = self._roulette_select(item_prob)

            bin_items[selected] = True
            vacancy -= self.demand[selected]
            valid[selected] = False
            path[selected] = bin_id
            item_count += 1

        vacancies.append(vacancy)
        return path, len(vacancies), path_fitness(vacancies, self.capacity)

    def _random_select(self, mask: IntArray) -> int:
        candidates = np.where(mask)[0]
        return candidates[floor(self.uniform_gen.next() * len(candidates))]

    def _roulette_select(self, prob: FloatArray) -> int:
        cumprob = np.cumsum(prob)
        r = self.uniform_gen.next() * cumprob[-1]
        return np.searchsorted(cumprob, r)

    def _update_pheromone(self, paths: List[IntArray], fitnesses: FloatArray):
        delta = np.zeros_like(self.pheromone)
        for path, f in zip(paths, fitnesses):
            same_bin = path[:, None] == path[None, :]
            delta[same_bin] += f / self.n_ants

        self.pheromone *= self.decay
        self.pheromone += delta

if __name__ == "__main__":
    demand = np.array([7, 4, 9, 3, 8, 5, 6, 2])
    capacity = 15

    solver = ACO_BPP_Solver(
        demand=demand,
        capacity=capacity,
        n_ants=40,
        iterations=200,
        seed=42,
    )

    num_bins, assignment = solver.solve()

    print("Number of bins:", num_bins)
    print("Assignment:", assignment)