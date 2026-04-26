import numpy as np
from typing import List, Tuple
import itertools

class HeldKarpTSP:
    """Held-Karp动态规划算法解决TSP
    适用于小规模问题（n <= 20），时间复杂度O(n²2ⁿ)"""
    
    def __init__(self, distance_matrix: np.ndarray):
        """
        初始化
        
        Args:
            distance_matrix: 距离矩阵，shape=(n, n)
        """
        self.n = len(distance_matrix)
        self.dist = distance_matrix
        
        if self.n > 20:  # 警告：动态规划只能处理小规模
            print(f"警告：Held-Karp算法处理 {self.n} 个城市需要 O(n²2ⁿ) 时间")
            print("建议使用启发式算法处理大规模问题")
    
    def solve(self) -> Tuple[List[int], float]:
        """
        求解TSP，返回最优路径和长度
        
        Returns:
            path: 最优路径（list of city indices）
            length: 最优路径长度
        """
        if self.n == 0:
            return [], 0.0
        if self.n == 1:
            return [0], 0.0
        
        # 初始化DP表
        dp, parent = self._initialize_dp()
        
        # 动态规划递推
        dp, parent = self._dp_iteration(dp, parent)
        
        # 回溯找到最优路径
        path, length = self._backtrack(dp, parent)
        
        return path, length
    
    def _initialize_dp(self) -> Tuple[np.ndarray, np.ndarray]:
        """初始化DP表和父节点表"""
        # dp[mask][i]: 访问mask中的城市，最后在i的最短路径
        dp = np.full((1 << self.n, self.n), np.inf)
        parent = np.full((1 << self.n, self.n), -1, dtype=int)
        
        # 初始化：从城市0开始
        dp[1][0] = 0  # mask=1 (二进制0001) 表示只访问了城市0
        
        return dp, parent
    
    def _dp_iteration(self, dp: np.ndarray, parent: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """动态规划递推"""
        # 遍历所有子集大小（从2个城市到n个城市）
        for subset_size in range(2, self.n + 1):
            # 生成所有大小为subset_size的子集（包含城市0）
            subsets = self._generate_subsets(subset_size)
            
            for mask in subsets:
                # 确保城市0在子集中（从0出发）
                if not (mask & 1):
                    continue
                
                # 遍历子集中的所有城市i（作为终点）
                for i in range(self.n):
                    if not (mask & (1 << i)):
                        continue
                    
                    # 如果i=0且不是完整的集合，跳过（不能提前回到0）
                    if i == 0 and mask != (1 << self.n) - 1:
                        continue
                    
                    # 尝试所有可能的j（i的前一个城市）
                    best_value = np.inf
                    best_j = -1
                    
                    for j in range(self.n):
                        if i == j or not (mask & (1 << j)):
                            continue
                        
                        # 状态转移
                        prev_mask = mask ^ (1 << i)  # 从mask中移除i
                        candidate = dp[prev_mask][j] + self.dist[j][i]
                        
                        if candidate < best_value:
                            best_value = candidate
                            best_j = j
                    
                    if best_j != -1:
                        dp[mask][i] = best_value
                        parent[mask][i] = best_j
        
        return dp, parent
    
    def _generate_subsets(self, size: int) -> List[int]:
        """生成所有包含城市0的大小为size的子集"""
        subsets = []
        
        # 先生成除了0以外的size-1个城市的组合
        other_cities = list(range(1, self.n))
        for combo in itertools.combinations(other_cities, size - 1):
            mask = 1  # 总是包含城市0
            for city in combo:
                mask |= (1 << city)
            subsets.append(mask)
        
        return subsets
    
    def _backtrack(self, dp: np.ndarray, parent: np.ndarray) -> Tuple[List[int], float]:
        """回溯找到最优路径"""
        full_mask = (1 << self.n) - 1  # 所有城市都访问过了
        
        # 找到回到城市0的最优路径
        min_length = np.inf
        last_city = -1
        
        for i in range(1, self.n):
            length = dp[full_mask][i] + self.dist[i][0]
            if length < min_length:
                min_length = length
                last_city = i
        
        if last_city == -1:
            return [], np.inf
        
        # 回溯路径
        path = []
        mask = full_mask
        current = last_city
        
        while current != -1:
            path.append(current)
            next_mask = mask ^ (1 << current)
            next_city = parent[mask][current]
            mask = next_mask
            current = next_city
        
        # 添加起点0，并反转路径
        path.append(0)
        path = path[::-1]
        
        return path, min_length
    
    def solve_with_path_reconstruction(self) -> Tuple[List[int], float, np.ndarray]:
        """求解并返回完整的DP表（用于分析）"""
        path, length = self.solve()
        
        # 重新计算DP表用于返回
        dp, _ = self._initialize_dp()
        
        for subset_size in range(2, self.n + 1):
            subsets = self._generate_subsets(subset_size)
            
            for mask in subsets:
                if not (mask & 1):
                    continue
                
                for i in range(self.n):
                    if not (mask & (1 << i)):
                        continue
                    
                    if i == 0 and mask != (1 << self.n) - 1:
                        continue
                    
                    best = np.inf
                    for j in range(self.n):
                        if i == j or not (mask & (1 << j)):
                            continue
                        
                        prev_mask = mask ^ (1 << i)
                        candidate = dp[prev_mask][j] + self.dist[j][i]
                        best = min(best, candidate)
                    
                    if best < np.inf:
                        dp[mask][i] = best
        
        return path, length, dp