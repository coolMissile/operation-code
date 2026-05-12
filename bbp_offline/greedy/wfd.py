from typing import List, Tuple

def organize_bins(assignment: List[int]) -> Tuple[int, List[int]]:
    """
    将箱子编号重新压缩为 0..k-1
    """
    mapping = {}
    new_assignment = []
    for b in assignment:
        if b not in mapping:
            mapping[b] = len(mapping)
        new_assignment.append(mapping[b])
    return len(mapping), new_assignment

class WorstFitDecreasing:
    def solve(self, weights: List[int], capacity: int) -> Tuple[int, List[int]]:
        weights = sorted(enumerate(weights), key=lambda x: -x[1])
        bins = []  # remaining space

        assignment = [-1] * len(weights)

        for idx, w in weights:
            worst_idx = -1
            worst_remain = -1

            for i, rem in enumerate(bins):
                if w <= rem and rem > worst_remain:
                    worst_remain = rem
                    worst_idx = i

            if worst_idx != -1:
                bins[worst_idx] -= w
                assignment[idx] = worst_idx
            else:
                bins.append(capacity - w)
                assignment[idx] = len(bins) - 1

        return len(bins), assignment