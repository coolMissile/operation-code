import numpy as np
import networkx as nx
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Set
import itertools

class ChristofidesTSP:
    """Christofides算法实现"""
    
    def __init__(self, distance_matrix: np.ndarray):
        """
        初始化
        
        Args:
            distance_matrix: 距离矩阵，必须满足三角不等式
        """
        self.dist = distance_matrix
        self.n = len(distance_matrix)
        
        # 验证三角不等式
        self._verify_triangle_inequality()
    
    def _verify_triangle_inequality(self):
        """验证三角不等式"""
        for i in range(self.n):
            for j in range(self.n):
                for k in range(self.n):
                    if self.dist[i][j] > self.dist[i][k] + self.dist[k][j] + 1e-9:
                        print(f"⚠️ 警告：不满足三角不等式 dist[{i}][{j}] > dist[{i}][{k}] + dist[{k}][{j}]")
                        print(f"  {self.dist[i][j]} > {self.dist[i][k]} + {self.dist[k][j]}")
                        return False
        return True
    
    def solve(self) -> Tuple[List[int], float]:
        """
        Christofides算法主函数
        
        Returns:
            tour: 哈密顿回路
            length: 回路长度
        """
        print("=" * 50)
        print("Christofides算法求解TSP")
        print("=" * 50)
        
        # 步骤1: 构建最小生成树
        print("\n1. 构建最小生成树...")
        mst_edges = self._construct_mst()
        print(f"   MST边数: {len(mst_edges)}")
        
        # 步骤2: 找到奇数度顶点
        print("\n2. 找到奇数度顶点...")
        odd_vertices = self._find_odd_degree_vertices(mst_edges)
        print(f"   奇数度顶点数量: {len(odd_vertices)}")
        print(f"   顶点: {odd_vertices}")
        
        # 步骤3: 最小权重完美匹配
        print("\n3. 最小权重完美匹配...")
        matching_edges = self._minimum_weight_perfect_matching(odd_vertices)
        print(f"   匹配边数: {len(matching_edges)}")
        
        # 步骤4: 合并MST和匹配
        print("\n4. 合并MST和匹配边...")
        eulerian_graph = self._combine_graphs(mst_edges, matching_edges)
        
        # 步骤5: 找到欧拉回路
        print("\n5. 找到欧拉回路...")
        eulerian_circuit = self._find_eulerian_circuit(eulerian_graph)
        print(f"   欧拉回路长度: {len(eulerian_circuit)}")
        
        # 步骤6: 转换为哈密顿回路（跳过重复顶点）
        print("\n6. 转换为哈密顿回路...")
        hamiltonian_tour = self._make_hamiltonian(eulerian_circuit)
        
        # 计算路径长度
        tour_length = self._calculate_tour_length(hamiltonian_tour)
        
        print(f"\n✅ 求解完成!")
        print(f"   路径长度: {tour_length:.2f}")
        print(f"   路径: {hamiltonian_tour}")
        
        return hamiltonian_tour, tour_length
    
    def _construct_mst(self) -> List[Tuple[int, int, float]]:
        """使用Prim算法构建最小生成树"""
        # 使用Prim算法
        visited = [False] * self.n
        mst_edges = []
        
        # 从顶点0开始
        visited[0] = True
        
        for _ in range(self.n - 1):
            min_edge = (None, None, float('inf'))
            
            # 在已访问和未访问顶点之间找最小边
            for u in range(self.n):
                if not visited[u]:
                    continue
                    
                for v in range(self.n):
                    if visited[v] or u == v:
                        continue
                    
                    weight = self.dist[u][v]
                    if weight < min_edge[2]:
                        min_edge = (u, v, weight)
            
            if min_edge[0] is not None:
                u, v, w = min_edge
                mst_edges.append((u, v, w))
                visited[v] = True
        
        return mst_edges
    
    def _find_odd_degree_vertices(self, edges: List[Tuple[int, int, float]]) -> List[int]:
        """在生成树中找到奇数度顶点"""
        degree = [0] * self.n
        
        for u, v, _ in edges:
            degree[u] += 1
            degree[v] += 1
        
        odd_vertices = [i for i, d in enumerate(degree) if d % 2 == 1]
        return odd_vertices
    
    def _minimum_weight_perfect_matching(self, vertices: List[int]) -> List[Tuple[int, int, float]]:
        """最小权重完美匹配（Blossom算法简化版）"""
        k = len(vertices)
        if k == 0:
            return []
        
        # 如果顶点数是奇数，理论上不会发生（在最小生成树中）
        if k % 2 == 1:
            print(f"警告：奇数个顶点({k})，无法完美匹配")
            # 添加一个虚拟顶点
            vertices.append(self.n)  # 虚拟顶点
        
        # 创建完全图
        m = len(vertices)
        cost_matrix = np.zeros((m, m))
        
        for i in range(m):
            for j in range(m):
                if i == j:
                    cost_matrix[i][j] = np.inf
                elif vertices[i] < self.n and vertices[j] < self.n:
                    cost_matrix[i][j] = self.dist[vertices[i]][vertices[j]]
                else:
                    cost_matrix[i][j] = 0  # 虚拟顶点距离为0
        
        # 使用匈牙利算法找到最小权重完美匹配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matching_edges = []
        matched = [False] * m
        
        for i, j in zip(row_ind, col_ind):
            if not matched[i] and not matched[j] and i < j:
                u, v = vertices[i], vertices[j]
                if u < self.n and v < self.n:  # 忽略虚拟顶点
                    weight = self.dist[u][v]
                    matching_edges.append((u, v, weight))
                matched[i] = matched[j] = True
        
        return matching_edges
    
    def _combine_graphs(self, mst_edges: List[Tuple], matching_edges: List[Tuple]) -> nx.MultiGraph:
        """合并MST和匹配边，构建欧拉图"""
        G = nx.MultiGraph()
        
        # 添加MST边
        for u, v, w in mst_edges:
            G.add_edge(u, v, weight=w)
        
        # 添加匹配边
        for u, v, w in matching_edges:
            G.add_edge(u, v, weight=w)
        
        return G
    
    def _find_eulerian_circuit(self, graph: nx.MultiGraph) -> List[int]:
        """使用Hierholzer算法找到欧拉回路"""
        if not nx.is_eulerian(graph):
            # 如果不满足欧拉图条件，返回一个简单路径
            return list(graph.nodes())
        
        # Hierholzer算法
        circuit = []
        stack = [0]  # 从顶点0开始
        current_graph = graph.copy()
        
        while stack:
            v = stack[-1]
            
            if current_graph.degree(v) > 0:
                # 找一条边
                for u in current_graph.neighbors(v):
                    stack.append(u)
                    # 删除边
                    current_graph.remove_edge(v, u)
                    break
            else:
                circuit.append(stack.pop())
        
        circuit.reverse()
        return circuit
    
    def _make_hamiltonian(self, eulerian_circuit: List[int]) -> List[int]:
        """从欧拉回路转换为哈密顿回路（跳过重复顶点）"""
        visited = set()
        hamiltonian = []
        
        for vertex in eulerian_circuit:
            if vertex not in visited:
                visited.add(vertex)
                hamiltonian.append(vertex)
        
        # 回到起点
        hamiltonian.append(hamiltonian[0])
        
        return hamiltonian
    
    def _calculate_tour_length(self, tour: List[int]) -> float:
        """计算回路长度"""
        total = 0.0
        for i in range(len(tour) - 1):
            total += self.dist[tour[i]][tour[i+1]]
        return total
    
    def analyze_approximation_ratio(self, optimal_length: float = None) -> dict:
        """分析近似比"""
        tour, length = self.solve()
        
        result = {
            'tour': tour,
            'length': length,
            'mst_length': self._calculate_mst_length(),
            'matching_length': self._calculate_matching_length(tour)
        }
        
        if optimal_length is not None:
            ratio = length / optimal_length
            result['approximation_ratio'] = ratio
            result['is_within_1.5'] = ratio <= 1.5 + 1e-9
        
        return result
    
    def _calculate_mst_length(self) -> float:
        """计算MST长度"""
        mst_edges = self._construct_mst()
        return sum(w for _, _, w in mst_edges)
    
    def _calculate_matching_length(self, tour: List[int]) -> float:
        """计算匹配边的长度"""
        odd_vertices = self._find_odd_degree_vertices(self._construct_mst())
        matching_edges = self._minimum_weight_perfect_matching(odd_vertices)
        return sum(w for _, _, w in matching_edges)