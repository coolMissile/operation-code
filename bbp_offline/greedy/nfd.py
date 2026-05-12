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
class NextFitDecreasing:
    def solve(self, weights: List[int], capacity: int) -> Tuple[int, List[int]]:
        weights = sorted(enumerate(weights), key=lambda x: -x[1])
        assignment = [-1] * len(weights)

        current_bin = 0
        remaining = capacity

        for idx, w in weights:
            if w <= remaining:
                assignment[idx] = current_bin
                remaining -= w
            else:
                current_bin += 1
                assignment[idx] = current_bin
                remaining = capacity - w

        return organize_bins(assignment)