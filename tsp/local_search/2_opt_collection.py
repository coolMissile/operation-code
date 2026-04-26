import numpy as np
from typing import List, Tuple, Optional
import time

class TwoOptOptimizer:
    """完整的2-opt优化工具类,包括4种不同的优化方法"""
    
    def __init__(self, distance_matrix: np.ndarray):
        """
        初始化2-opt优化器
        
        Args:
            distance_matrix: 距离矩阵
        """
        self.dist = distance_matrix
        self.n = len(distance_matrix)
        
    def optimize(self, initial_tour: List[int], 
                 method: str = 'standard',
                 max_time: float = 10.0,
                 max_iter: int = 1000,
                 candidate_size: int = 20) -> Tuple[List[int], float, dict]:
        """
        2-opt优化
        
        Args:
            initial_tour: 初始路径
            method: 优化方法 ('standard', 'fast', 'candidate', 'lkh')
            max_time: 最大运行时间(秒)
            max_iter: 最大迭代次数
            candidate_size: 候选集大小（仅method='candidate'时使用）
        
        Returns:
            best_tour: 优化后的路径
            best_len: 路径长度
            stats: 优化统计信息
        """
        start_time = time.time()
        
        if method == 'standard':
            best_tour, best_len = self._standard_two_opt(
                initial_tour, max_iter, start_time, max_time
            )
        elif method == 'fast':
            best_tour, best_len = self._fast_two_opt(
                initial_tour, max_iter, start_time, max_time
            )
        elif method == 'candidate':
            best_tour, best_len = self._candidate_two_opt(
                initial_tour, candidate_size, start_time, max_time
            )
        elif method == 'lkh':
            best_tour, best_len, _ = self._lkh_two_opt(
                initial_tour, start_time, max_time
            )
        else:
            raise ValueError(f"未知的优化方法: {method}")
        
        stats = {
            'initial_length': self._tour_length(initial_tour),
            'final_length': best_len,
            'improvement': self._tour_length(initial_tour) - best_len,
            'improvement_percent': 100 * (1 - best_len / self._tour_length(initial_tour)),
            'time_elapsed': time.time() - start_time
        }
        
        return best_tour, best_len, stats
    
    def _standard_two_opt(self, tour: List[int], max_iter: int,
                         start_time: float, max_time: float) -> Tuple[List[int], float]:
        """标准2-opt实现"""
        n = self.n
        best_tour = tour.copy()
        best_len = self._tour_length(tour)
        
        improved = True
        iteration = 0
        
        while improved and iteration < max_iter and \
              time.time() - start_time < max_time:
            
            improved = False
            
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if j == n - 1 and i == 0:
                        continue
                    
                    a, b = best_tour[i], best_tour[i + 1]
                    c, d = best_tour[j], best_tour[(j + 1) % n]
                    
                    delta = (self.dist[a][b] + self.dist[c][d] -
                            (self.dist[a][c] + self.dist[b][d]))
                    
                    if delta > 1e-9:  # 有改进
                        if i + 1 < j:
                            best_tour[i+1:j+1] = best_tour[i+1:j+1][::-1]
                        else:
                            # 处理跨越起点
                            segment = best_tour[j:i:-1]
                            best_tour = segment + best_tour[i+1:j]
                        
                        best_len -= delta
                        improved = True
                        break
                
                if improved:
                    break
            
            iteration += 1
        
        return best_tour, best_len
    
    def _fast_two_opt(self, tour: List[int], max_iter: int,
                     start_time: float, max_time: float) -> Tuple[List[int], float]:
        """快速2-opt实现"""
        n = self.n
        best_tour = tour.copy()
        best_len = self._tour_length(tour)
        
        # 创建位置映射
        position = {city: idx for idx, city in enumerate(best_tour)}
        
        improved = True
        iteration = 0
        
        while improved and iteration < max_iter and \
              time.time() - start_time < max_time:
            
            improved = False
            
            for i in range(n):
                city_i = best_tour[i]
                city_i1 = best_tour[(i + 1) % n]
                
                for j in range(i + 2, n + i - 1):
                    j_mod = j % n
                    if (j_mod + 1) % n == i:
                        continue
                    
                    city_j = best_tour[j_mod]
                    city_j1 = best_tour[(j_mod + 1) % n]
                    
                    delta = (self.dist[city_i][city_j] +
                            self.dist[city_i1][city_j1] -
                            self.dist[city_i][city_i1] -
                            self.dist[city_j][city_j1])
                    
                    if delta < -1e-9:  # 有改进
                        # 执行交换
                        if i < j_mod:
                            best_tour[i+1:j_mod+1] = best_tour[i+1:j_mod+1][::-1]
                        else:
                            segment = best_tour[j_mod:i:-1]
                            best_tour = segment + best_tour[i+1:j_mod]
                        
                        best_len += delta
                        improved = True
                        
                        # 更新位置映射
                        for idx, city in enumerate(best_tour):
                            position[city] = idx
                        
                        break
                
                if improved:
                    break
            
            iteration += 1
        
        return best_tour, best_len
    
    def _candidate_two_opt(self, tour: List[int], candidate_size: int,
                          start_time: float, max_time: float) -> Tuple[List[int], float]:
        """候选集2-opt"""
        n = self.n
        
        # 构建候选集
        candidate_lists = []
        for i in range(n):
            distances = self.dist[i].copy()
            distances[i] = np.inf
            candidates = np.argsort(distances)[:candidate_size]
            candidate_lists.append(set(candidates))
        
        best_tour = tour.copy()
        best_len = self._tour_length(tour)
        
        # 位置映射
        position = {city: idx for idx, city in enumerate(best_tour)}
        
        improved = True
        
        while improved and time.time() - start_time < max_time:
            improved = False
            
            for i in range(n):
                city_i = best_tour[i]
                city_i1 = best_tour[(i + 1) % n]
                
                # 只检查候选邻居
                for city_j in candidate_lists[city_i1]:
                    j = position[city_j]
                    j_next = (j + 1) % n
                    city_j1 = best_tour[j_next]
                    
                    if (j_next) % n == i or j == i:
                        continue
                    
                    delta = (self.dist[city_i][city_j] +
                            self.dist[city_i1][city_j1] -
                            self.dist[city_i][city_i1] -
                            self.dist[city_j][city_j1])
                    
                    if delta < -1e-9:
                        if i < j:
                            best_tour[i+1:j+1] = best_tour[i+1:j+1][::-1]
                        else:
                            segment = best_tour[j:i:-1]
                            best_tour = segment + best_tour[i+1:j]
                        
                        best_len += delta
                        improved = True
                        
                        # 更新位置
                        for idx, city in enumerate(best_tour):
                            position[city] = idx
                        
                        break
                
                if improved:
                    break
        
        return best_tour, best_len
    
    def _lkh_two_opt(self, tour: List[int], 
                     start_time: float, max_time: float) -> Tuple[List[int], float, list]:
        """LKH风格的2-opt"""
        n = self.n
        current_tour = tour.copy()
        current_len = self._tour_length(tour)
        
        # 邻接表
        next_node = [0] * n
        prev_node = [0] * n
        
        for i in range(n):
            next_node[current_tour[i]] = current_tour[(i + 1) % n]
            prev_node[current_tour[i]] = current_tour[(i - 1) % n]
        
        swaps = []
        improved = True
        
        while improved and time.time() - start_time < max_time:
            improved = False
            best_gain = 0.0
            best_swap = None
            
            # 找到最佳交换
            for i in range(n):
                t1 = current_tour[i]
                t2 = next_node[t1]
                
                for j in range(i + 1, n):
                    t3 = current_tour[j]
                    t4 = next_node[t3]
                    
                    if t2 == t3 or t4 == t1:
                        continue
                    
                    deleted = self.dist[t1][t2] + self.dist[t3][t4]
                    added = self.dist[t1][t3] + self.dist[t2][t4]
                    gain = deleted - added
                    
                    if gain > best_gain + 1e-9:
                        best_gain = gain
                        best_swap = (i, j, t1, t2, t3, t4)
            
            # 执行交换
            if best_swap is not None and best_gain > 1e-9:
                i, j, t1, t2, t3, t4 = best_swap
                
                current_len -= best_gain
                swaps.append((i, j))
                
                # 更新邻接表
                next_node[t1] = t3
                prev_node[t3] = t1
                next_node[t2] = t4
                prev_node[t4] = t2
                
                # 反转t2到t3之间的段
                current = t2
                while current != t3:
                    next_temp = next_node[current]
                    prev_temp = prev_node[current]
                    next_node[current] = prev_temp
                    prev_node[current] = next_temp
                    current = next_temp
                
                # 重建路径
                new_tour = [0] * n
                current = 0
                for k in range(n):
                    new_tour[k] = current
                    current = next_node[current]
                
                current_tour = new_tour
                improved = True
        
        return current_tour, current_len, swaps
    
    def _tour_length(self, tour: List[int]) -> float:
        """计算路径长度"""
        total = 0.0
        n = len(tour)
        for i in range(n):
            j = (i + 1) % n
            total += self.dist[tour[i]][tour[j]]
        return total