import numpy as np
import networkx as nx
from typing import List, Tuple, Set
import itertools
import heapq


class ChristofidesTSP:
    """Christofides算法实现"""
    
    def __init__(self, distance_matrix: np.ndarray):
        """
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
                        print(f"警告：不满足三角不等式 dist[{i}][{j}] > dist[{i}][{k}] + dist[{k}][{j}]")
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
        
        print(f"\n求解完成!")
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
        """贪心近似的最小权重完美匹配（无SciPy依赖）"""
        k = len(vertices)
        if k < 2:
            return []
        
        # 对顶点按距离排序，使用贪心策略
        unpaired = set(vertices)
        matching_edges = []
        
        while unpaired:
            u = unpaired.pop()
            
            # 找到最近的未匹配顶点
            min_dist = float('inf')
            closest = -1
            
            for v in unpaired:
                dist = self.dist[u][v]
                if dist < min_dist:
                    min_dist = dist
                    closest = v
            
            if closest != -1:
                matching_edges.append((u, closest, min_dist))
                unpaired.remove(closest)
            else:
                # 如果没有可匹配的顶点，随机连接
                for v in range(self.n):
                    if v != u and v not in unpaired:
                        matching_edges.append((u, v, self.dist[u][v]))
                        break
        
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
 
if __name__ == "__main__":
    # Example 1: 使用随机生成的满足三角不等式的距离矩阵
    print("=" * 60)
    print("测试1: 随机生成的距离矩阵")
    print("=" * 60)
    
    N = 10
    np.random.seed(42)  
    
    # 生成随机点坐标
    points = np.random.rand(N, 2) * 100
    
    # 计算欧几里得距离（自动满足三角不等式）
    dist_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                dist_matrix[i][j] = np.linalg.norm(points[i] - points[j])
            else:
                dist_matrix[i][j] = 0
    
    print(f"生成了 {N} 个随机点")
    print(f"距离矩阵形状: {dist_matrix.shape}")
    
    # 创建Christofides求解器
    solver = ChristofidesTSP(dist_matrix)
    
    # 求解TSP
    tour, length = solver.solve()
    
    # 分析近似比（这里用最近邻算法作为粗略的"最优"参考）
    print("\n" + "=" * 60)
    print("近似比分析:")
    print("=" * 60)
    
    # 使用最近邻算法得到参考解
    def nearest_neighbor(dist_matrix):
        n = len(dist_matrix)
        unvisited = set(range(n))
        current = 0
        tour = [current]
        unvisited.remove(current)
        total_length = 0
        
        while unvisited:
            # 找到最近的未访问城市
            nearest = min(unvisited, key=lambda x: dist_matrix[current][x])
            total_length += dist_matrix[current][nearest]
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        # 回到起点
        total_length += dist_matrix[tour[-1]][tour[0]]
        tour.append(tour[0])
        
        return tour, total_length
    
    nn_tour, nn_length = nearest_neighbor(dist_matrix)
    print(f"最近邻算法解长度: {nn_length:.4f}")
    print(f"Christofides算法解长度: {length:.4f}")
    
    if length <= nn_length:
        ratio = length / nn_length if nn_length > 0 else 1
        print(f"Christofides优于最近邻: 比例为 {ratio:.4f}")
    else:
        ratio = length / nn_length
        print(f"最近邻优于Christofides: 比例为 {ratio:.4f}")
    
    print("\n" + "=" * 60)
    print("测试2: 使用给定的距离矩阵格式")
    print("=" * 60)
    
    # 按照你提供的格式生成距离矩阵
    N2 = 8
    distances2 = []
    for i in range(N2):
        row = [np.random.random() for _ in range(i)]
        distances2.append(row)
    
    # 转换为完整距离矩阵
    full_dist2 = np.zeros((N2, N2))
    for i in range(N2):
        for j in range(i):
            full_dist2[i][j] = distances2[i][j]
            full_dist2[j][i] = distances2[i][j]
    
    print(f"生成 {N2} 个点的距离矩阵")
    print("距离矩阵（上三角）:")
    for i in range(min(N2, 5)):  # 只显示前5行
        print(f"  {i}: {distances2[i]}")
    
    # 由于随机生成的距离可能不满足三角不等式，我们需要处理
    print("\n检查三角不等式...")
    solver2 = ChristofidesTSP(full_dist2)
    
    # 如果通过了检查，尝试求解
    try:
        tour2, length2 = solver2.solve()
        print(f"找到路径: {tour2}")
        print(f"路径长度: {length2:.4f}")
    except Exception as e:
        print(f"求解失败: {e}")
        print("随机生成的距离矩阵可能不满足三角不等式")
        print("建议使用欧几里得距离或满足三角不等式的距离")
    
    print("\n" + "=" * 60)
    print("测试3: 小规模示例（满足三角不等式）")
    print("=" * 60)
    
    # 创建一个满足三角不等式的小例子
    # 4个点形成正方形
    points3 = np.array([
        [0, 0],  # 点0
        [0, 1],  # 点1
        [1, 0],  # 点2
        [1, 1]   # 点3
    ])
    
    N3 = 4
    dist_matrix3 = np.zeros((N3, N3))
    for i in range(N3):
        for j in range(N3):
            if i != j:
                dist_matrix3[i][j] = np.linalg.norm(points3[i] - points3[j])
    
    print("正方形四点问题:")
    print("点坐标:")
    for i, (x, y) in enumerate(points3):
        print(f"  点{i}: ({x}, {y})")
    
    print("\n距离矩阵:")
    print(dist_matrix3)
    
    solver3 = ChristofidesTSP(dist_matrix3)
    tour3, length3 = solver3.solve()
    
    # 计算最优解（手动计算）
    optimal_tour = [0, 1, 3, 2, 0]  
    optimal_length = 0
    for i in range(len(optimal_tour) - 1):
        optimal_length += dist_matrix3[optimal_tour[i]][optimal_tour[i+1]]
    
    print(f"\n最优解长度: {optimal_length:.4f}")
    print(f"Christofides解长度: {length3:.4f}")
    print(f"近似比: {length3/optimal_length:.4f}")
    
    if length3/optimal_length <= 1.5 + 1e-9:
        print("满足1.5倍近似比保证")
    else:
        print("不满足1.5倍近似比保证")