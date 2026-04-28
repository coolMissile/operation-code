#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
独立运行的最近邻CVRP求解器
支持多种数据格式：TSPLIB、标准CVRP、自定义坐标
由https://github.com/yorak/VeRyPy.git改编
"""

import numpy as np
import argparse
import sys
import os
from math import sqrt
import time

class CVRPProblem:
    def __init__(self, name, dimension, capacity, demands, coordinates=None, distances=None):
        self.name = name
        self.dimension = dimension
        self.capacity = capacity
        self.demands = demands
        self.coordinates = coordinates
        self.distances = distances
        if distances is None and coordinates is not None:
            self._compute_distance_matrix()
    
    def _compute_distance_matrix(self):
        """从坐标计算欧几里得距离矩阵"""
        n = self.dimension
        self.distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = self.coordinates[i][0] - self.coordinates[j][0]
                    dy = self.coordinates[i][1] - self.coordinates[j][1]
                    self.distances[i][j] = round(sqrt(dx*dx + dy*dy))

class _PeekQueue:
    """允许查看但不移除元素的自定义队列"""
    def __init__(self, items):
        self.posleft = -1
        self.posright = 0
        self.items = list(items)
    
    def __len__(self):
        return len(self.items) - (self.posleft + 1) + self.posright
    
    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError
        return self.items[self.posleft + 1 + idx]
    
    def peekleft(self):
        if len(self) == 0:
            raise IndexError
        return self.items[self.posleft + 1]
    
    def popleft(self):
        if len(self) == 0:
            raise IndexError
        self.posleft += 1
        return self.items[self.posleft]

class NearestNeighborSolver:
    """最近邻CVRP求解器"""
    
    def __init__(self, parallel_routes=1, seed_method='farthest', 
                 insert_at_end=False, verbose=False):
        """
        参数:
        - parallel_routes: 并行构建的路径数 (1=串行, >1=并行)
        - seed_method: 初始客户点选择方法 ('farthest', 'closest', 'nearest')
        - insert_at_end: 是否只能插入路径末端
        - verbose: 是否显示详细求解过程
        """
        self.parallel_routes = parallel_routes
        self.seed_method = seed_method
        self.insert_at_end = insert_at_end
        self.verbose = verbose
        
    def solve(self, problem):
        """求解CVRP问题，返回路径列表和总距离"""
        D = problem.distances
        d = problem.demands
        C = problem.capacity
        
        N = len(D)
        
        # 1. 为每个客户构建最近邻列表
        node_nearest_neighbors = [None] * N
        for i in range(N):
            # 按距离排序，排除自己
            neighbors = [(j, D[i][j]) for j in range(N) if j != i]
            neighbors.sort(key=lambda x: x[1])
            node_nearest_neighbors[i] = _PeekQueue(neighbors)
        
        # 2. 记录已服务客户
        served = [False] * N
        served[0] = True  # 仓库
        
        # 3. 初始化路径
        routes = []
        current_routes = [None] * self.parallel_routes
        route_demands = [0.0] * self.parallel_routes
        route_costs = [0.0] * self.parallel_routes
        
        # 工具函数：获取初始客户点
        def get_seed_node():
            if self.seed_method == 'farthest':
                # 找离仓库最远的未服务客户
                farthest_dist = -1
                seed = None
                for i in range(1, N):
                    if not served[i] and D[0][i] > farthest_dist:
                        farthest_dist = D[0][i]
                        seed = i
                return seed
                
            elif self.seed_method == 'closest':
                # 找离仓库最近的未服务客户
                closest_dist = float('inf')
                seed = None
                for i in range(1, N):
                    if not served[i] and D[0][i] < closest_dist:
                        closest_dist = D[0][i]
                        seed = i
                return seed
                
            elif self.seed_method == 'nearest':
                # 找相互最近的未服务客户对
                min_pair_dist = float('inf')
                seed = None
                for i in range(1, N):
                    if served[i]:
                        continue
                    # 找到i的最近未服务邻居
                    for j in range(len(node_nearest_neighbors[i])):
                        neighbor, dist = node_nearest_neighbors[i][j]
                        if not served[neighbor]:
                            if dist < min_pair_dist:
                                min_pair_dist = dist
                                seed = i
                            break
                return seed if seed is not None else 1
            else:
                raise ValueError(f"不支持的seed_method: {self.seed_method}")
        
        # 4. 主求解循环
        k = 0
        while not all(served):
            if self.verbose:
                remaining = sum(1 for s in served if not s)
                print(f"\n剩余未服务客户: {remaining}")
            
            # 如果当前路径为空，初始化新路径
            if current_routes[k] is None:
                seed = get_seed_node()
                if seed is None:  # 所有客户都已服务
                    break
                    
                current_routes[k] = [seed]
                route_demands[k] = d[seed]
                served[seed] = True
                
                if self.verbose:
                    print(f"  路径{k}: 初始客户 {seed}, 需求 {d[seed]}/{C}")
                
            else:
                # 找到要插入的客户
                route = current_routes[k]
                first = route[0]
                last = route[-1]
                
                # 查找最近的未服务客户
                best_customer = None
                best_position = None  # 0=前端, 1=末端
                best_increase = float('inf')
                
                # 检查路径前端的插入
                if not self.insert_at_end:
                    for i in range(1, N):
                        if not served[i]:
                            # 插入前端的距离增加: depot->i + i->first - depot->first
                            increase = D[0][i] + D[i][first] - D[0][first]
                            if increase < best_increase:
                                best_increase = increase
                                best_customer = i
                                best_position = 0
                
                # 检查路径末端的插入
                for i in range(1, N):
                    if not served[i]:
                        # 插入末端的距离增加: last->i + i->depot - last->depot
                        increase = D[last][i] + D[i][0] - D[last][0]
                        if increase < best_increase:
                            best_increase = increase
                            best_customer = i
                            best_position = 1
                
                if best_customer is None:
                    # 没有可插入的客户，关闭当前路径
                    routes.append([0] + current_routes[k] + [0])
                    current_routes[k] = None
                    continue
                
                # 检查容量约束
                if route_demands[k] + d[best_customer] > C + 1e-6:
                    # 容量不足，关闭当前路径
                    routes.append([0] + current_routes[k] + [0])
                    current_routes[k] = None
                    if self.verbose:
                        cost = D[0][first] + route_costs[k] + D[last][0]
                        print(f"  路径{k}: 容量不足，关闭路径，成本 {cost:.2f}")
                    continue
                
                # 插入客户
                if best_position == 0:  # 插入前端
                    current_routes[k].insert(0, best_customer)
                    if len(route) > 0:
                        route_costs[k] += D[best_customer][first]
                else:  # 插入末端
                    current_routes[k].append(best_customer)
                    if len(route) > 0:
                        route_costs[k] += D[last][best_customer]
                
                route_demands[k] += d[best_customer]
                served[best_customer] = True
                
                if self.verbose:
                    pos = "前端" if best_position == 0 else "末端"
                    print(f"  路径{k}: 在{pos}插入客户 {best_customer}, " +
                          f"需求 {route_demands[k]:.1f}/{C}, " +
                          f"距离增加 {best_increase:.2f}")
            
            # 轮转到下一条路径（并行版本）
            k = (k + 1) % self.parallel_routes
        
        # 5. 关闭所有剩余路径
        for i in range(self.parallel_routes):
            if current_routes[i] is not None and len(current_routes[i]) > 0:
                routes.append([0] + current_routes[i] + [0])
        
        return routes
    
    def calculate_total_distance(self, routes, distance_matrix):
        """计算总行驶距离"""
        total = 0
        for route in routes:
            for i in range(len(route) - 1):
                total += distance_matrix[route[i]][route[i+1]]
        return total
    
    def calculate_route_stats(self, routes, demands, capacity):
        """计算路径统计信息"""
        stats = []
        for idx, route in enumerate(routes):
            route_demand = sum(demands[node] for node in route)
            utilization = (route_demand / capacity) * 100
            num_customers = len(route) - 2  # 排除仓库
            stats.append({
                'route_id': idx + 1,
                'customers': route[1:-1],  # 去掉首尾的0
                'num_customers': num_customers,
                'demand': route_demand,
                'capacity': capacity,
                'utilization': utilization
            })
        return stats

def load_cvrp_instance(filepath):
    """加载CVRP实例文件"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # 解析基本信息
    name = None
    dimension = 0
    capacity = 0
    coordinates = []
    demands = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('NAME'):
            name = line.split(':')[1].strip()
        elif line.startswith('DIMENSION'):
            dimension = int(line.split(':')[1].strip())
        elif line.startswith('CAPACITY'):
            capacity = int(line.split(':')[1].strip())
        elif line.startswith('NODE_COORD_SECTION'):
            i += 1
            for j in range(dimension):
                parts = lines[i].strip().split()
                if len(parts) >= 3:
                    idx = int(parts[0]) - 1
                    x = float(parts[1])
                    y = float(parts[2])
                    coordinates.append((x, y))
                i += 1
            continue
        elif line.startswith('DEMAND_SECTION'):
            i += 1
            demands = [0] * dimension
            for j in range(dimension):
                parts = lines[i].strip().split()
                if len(parts) >= 2:
                    idx = int(parts[0]) - 1
                    demand = int(parts[1])
                    demands[idx] = demand
                i += 1
            continue
        elif line == 'EOF':
            break
        i += 1
    
    # 创建问题实例
    problem = CVRPProblem(
        name=name or os.path.basename(filepath),
        dimension=dimension,
        capacity=capacity,
        demands=demands,
        coordinates=coordinates
    )
    
    return problem

def create_random_instance(num_customers=20, grid_size=100, capacity=100, seed=42):
    """创建随机CVRP实例"""
    np.random.seed(seed)
    
    # 生成坐标
    coordinates = [(0, 0)]  # 仓库在原点
    for _ in range(num_customers):
        x = np.random.randint(1, grid_size)
        y = np.random.randint(1, grid_size)
        coordinates.append((x, y))
    
    # 生成需求
    demands = [0]  # 仓库需求为0
    for _ in range(num_customers):
        demands.append(np.random.randint(5, 30))
    
    problem = CVRPProblem(
        name=f"random_{num_customers}c",
        dimension=num_customers + 1,
        capacity=capacity,
        demands=demands,
        coordinates=coordinates
    )
    
    return problem

def print_solution(problem, routes, total_distance, stats, verbose=True):
    """打印解决方案"""
    print(f"\n{'='*60}")
    print(f"问题: {problem.name}")
    print(f"客户数: {problem.dimension - 1}")
    print(f"车辆容量: {problem.capacity}")
    print(f"使用车辆: {len(routes)}")
    print(f"总行驶距离: {total_distance:.2f}")
    print(f"{'='*60}")
    
    if verbose:
        print("\n路径详情:")
        for route_stat in stats:
            print(f"\n路径 {route_stat['route_id']}:")
            route_str = "0 -> " + " -> ".join(str(c) for c in route_stat['customers']) + " -> 0"
            print(f"  路径: {route_str}")
            print(f"  客户数: {route_stat['num_customers']}")
            print(f"  总需求: {route_stat['demand']:.1f} / {route_stat['capacity']} " +
                  f"({route_stat['utilization']:.1f}%)")
    
    # 简单格式输出（便于复制）
    print(f"\n解格式 (0表示仓库):")
    for i, route in enumerate(routes):
        print(f"路径{i+1}: {' '.join(str(node) for node in route)}")

# def save_solution(filename, problem, routes, total_distance):
#     """保存解决方案到文件"""
#     with open(filename, 'w') as f:
#         f.write(f"NAME: {problem.name}_solution\n")
#         f.write(f"COMMENT: Nearest Neighbor Solution\n")
#         f.write(f"TYPE: CVRP\n")
#         f.write(f"DIMENSION: {problem.dimension}\n")
#         f.write(f"VEHICLES: {len(routes)}\n")
#         f.write(f"TOTAL_DISTANCE: {total_distance:.2f}\n")
#         f.write("ROUTES_SECTION\n")
        
#         for i, route in enumerate(routes):
#             f.write(f"Route {i+1}: {' '.join(str(node) for node in route)}\n")
        
#         f.write("EOF\n")

def main():
    parser = argparse.ArgumentParser(description='最近邻CVRP求解器')
    parser.add_argument('input', nargs='?', help='输入文件路径 (TSPLIB格式)')
    parser.add_argument('-o', '--output', help='输出文件路径')
    parser.add_argument('-r', '--random', type=int, help='生成随机问题 (客户数)')
    parser.add_argument('-p', '--parallel', type=int, default=1, 
                       help='并行构建的路径数 (默认: 1)')
    parser.add_argument('-s', '--seed', choices=['farthest', 'closest', 'nearest'], 
                       default='farthest', help='初始客户选择方法')
    parser.add_argument('-e', '--end-only', action='store_true',
                       help='只能插入路径末端')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='显示详细求解过程')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='只显示结果')
    
    args = parser.parse_args()
    
    # 检查输入
    if not args.input and not args.random:
        print("错误: 需要提供输入文件或使用--random生成随机问题")
        parser.print_help()
        sys.exit(1)
    
    # 加载或生成问题
    if args.random:
        print(f"生成随机CVRP问题: {args.random}个客户")
        problem = create_random_instance(num_customers=args.random)
    else:
        print(f"加载问题: {args.input}")
        try:
            problem = load_cvrp_instance(args.input)
        except Exception as e:
            print(f"错误: 无法加载文件: {e}")
            sys.exit(1)
    
    if not args.quiet:
        print(f"问题: {problem.name}")
        print(f"客户数: {problem.dimension - 1}")
        print(f"车辆容量: {problem.capacity}")
        if problem.coordinates is not None:
            print(f"坐标范围: 0-{max(max(c) for c in problem.coordinates)}")
   
    solver = NearestNeighborSolver(
        parallel_routes=args.parallel,
        seed_method=args.seed,
        insert_at_end=args.end_only,
        verbose=args.verbose and not args.quiet
    )
    
    start_time = time.time()
    routes = solver.solve(problem)
    solve_time = time.time() - start_time
    
    if routes is None or len(routes) == 0:
        print("错误: 无法找到可行解")
        sys.exit(1)
    
    # 计算结果
    total_distance = solver.calculate_total_distance(routes, problem.distances)
    stats = solver.calculate_route_stats(routes, problem.demands, problem.capacity)
    
    # 输出结果
    if not args.quiet:
        print_solution(problem, routes, total_distance, stats, verbose=not args.quiet)
        print(f"\n求解时间: {solve_time:.3f}秒")
    
    # 保存结果
    # if args.output:
    #     save_solution(args.output, problem, routes, total_distance)
    #     if not args.quiet:
    #         print(f"解已保存到: {args.output}")
    
    return routes, total_distance

if __name__ == "__main__":
    main()