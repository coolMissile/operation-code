import numpy as np
from typing import List, Set

def nearest_neighbor_tsp(node_positions: np.ndarray) -> float:
    """
    最近邻算法解决TSP
    
    Parameters
    ----------
    node_positions : np.ndarray
        节点坐标，shape=(n, 2)
    
    Returns
    -------
    tour_length : float
        路径长度
    """
    n = node_positions.shape[0]
    
    # 计算距离矩阵
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_mat[i, j] = np.linalg.norm(node_positions[i] - node_positions[j])
            else:
                dist_mat[i, j] = np.inf
    
    #随机起点
    start = np.random.randint(0, n)
    tour = [start]
    unvisited = set(range(n))
    unvisited.remove(start)
    
    #最近邻贪心
    for _ in range(n - 1):
        current = tour[-1]
        min_dist = np.inf
        nearest = -1
        
        for city in unvisited:
            if dist_mat[current, city] < min_dist:
                min_dist = dist_mat[current, city]
                nearest = city
        
        tour.append(nearest)
        unvisited.remove(nearest)
    
    #计算路径长度
    total_length = 0
    for i in range(n):
        j = (i + 1) % n
        total_length += dist_mat[tour[i], tour[j]]
    
    return total_length, tour