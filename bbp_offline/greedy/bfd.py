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

class BestFitDecreasing:
    def solve(self, weights: List[int], capacity: int) -> Tuple[int, List[int]]:
        weights = sorted(enumerate(weights), key=lambda x: -x[1])
        bins = []  # remaining space

        assignment = [-1] * len(weights)

        for idx, w in weights:
            best_idx = -1
            best_remain = capacity + 1

            for i, rem in enumerate(bins):
                if w <= rem < best_remain:
                    best_remain = rem
                    best_idx = i

            if best_idx != -1:
                bins[best_idx] -= w
                assignment[idx] = best_idx
            else:
                bins.append(capacity - w)
                assignment[idx] = len(bins) - 1

        return len(bins), assignment