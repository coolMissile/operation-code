#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Branch and Bound算法求解CVRP（车辆路径问题）"""

import numpy as np
import math
import heapq
import time
from itertools import permutations, combinations
import matplotlib.pyplot as plt

class Node:
    """分支定界树中的节点"""
    def __init__(self, level, path, cost, demand, lower_bound, excluded_edges=set()):
        self.level = level              # 节点在树中的深度
        self.path = path[:]            # 当前部分路径
        self.cost = cost              # 当前路径的成本
        self.demand = demand          # 当前路径的总需求
        self.lower_bound = lower_bound  # 下界估计
        self.excluded_edges = excluded_edges  # 禁止使用的边
        
    def __lt__(self, other):
        # 优先队列按lower_bound排序
        return self.lower_bound < other.lower_bound

def calculate_distance_matrix(points):
    """计算欧氏距离矩阵"""
    n = len(points)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            D[i, j] = math.sqrt(dx*dx + dy*dy)
    return D

def nearest_neighbor_lower_bound(D, unvisited_nodes, current_node, depot=0):
    """最近邻下界：估计完成当前路径的最小成本"""
    if not unvisited_nodes:
        return D[current_node, depot]
    
    # 找到最近的未访问节点
    min_dist = float('inf')
    for node in unvisited_nodes:
        if D[current_node, node] < min_dist:
            min_dist = D[current_node, node]
    
    return min_dist

def mst_lower_bound(D, nodes):
    """最小生成树下界"""
    if not nodes:
        return 0
    
    # Prim算法计算MST
    visited = {nodes[0]}
    mst_cost = 0
    
    while len(visited) < len(nodes):
        min_edge = float('inf')
        min_node = None
        
        for u in visited:
            for v in nodes:
                if v not in visited and D[u, v] < min_edge:
                    min_edge = D[u, v]
                    min_node = v
        
        if min_node is not None:
            visited.add(min_node)
            mst_cost += min_edge
    
    return mst_cost

def calculate_lower_bound(D, current_path, unvisited_nodes, current_cost, capacity, 
                         vehicle_capacity, depot=0):
    """计算节点的下界估计"""
    if not unvisited_nodes:
        return current_cost + D[current_path[-1], depot]
    
    # 1. 当前路径的剩余成本
    lower_bound = current_cost
    
    # 2. 从最后一个节点到最近未访问节点的距离
    last_node = current_path[-1]
    if unvisited_nodes:
        min_to_next = min(D[last_node, node] for node in unvisited_nodes)
        lower_bound += min_to_next
    
    # 3. 从未访问节点返回仓库的最小距离
    min_to_depot = min(D[node, depot] for node in unvisited_nodes)
    lower_bound += min_to_depot
    
    # 4. 考虑最小生成树下界（可选，更紧但更慢）
    # mst_bound = mst_lower_bound(D, list(unvisited_nodes) + [depot])
    # lower_bound = max(lower_bound, current_cost + mst_bound)
    
    return lower_bound

def branch_and_bound_cvrp(D, demands, vehicle_capacity, time_limit=30):
    """
    Branch and Bound算法求解CVRP
    
    参数:
    D: 距离矩阵
    demands: 需求列表，demands[0]是仓库需求(0)
    vehicle_capacity: 车辆容量
    time_limit: 时间限制（秒）
    
    返回:
    best_solution: 最优解
    best_cost: 最优成本
    nodes_explored: 探索的节点数
    """
    n = len(D)  # 包括仓库
    depot = 0
    
    print("=" * 60)
    print("Branch and Bound算法求解CVRP")
    print("=" * 60)
    print(f"节点数: {n-1} (不包括仓库)")
    print(f"车辆容量: {vehicle_capacity}")
    print(f"总需求: {sum(demands)}")
    print(f"最小车辆数: {math.ceil(sum(demands) / vehicle_capacity)}")
    
    # 初始化最优解
    best_solution = None
    best_cost = float('inf')
    nodes_explored = 0
    pruned_nodes = 0
    
    # 优先队列（最小堆），按lower_bound排序
    priority_queue = []
    
    # 创建根节点
    start_time = time.time()
    
    # 初始下界估计（使用最近邻贪心解）
    initial_lower_bound = 0
    unvisited = set(range(1, n))
    current = depot
    temp_demand = 0
    temp_path = [depot]
    
    # 贪心构造初始解
    while unvisited:
        # 找到最近的可达节点
        nearest = None
        min_dist = float('inf')
        
        for node in unvisited:
            if temp_demand + demands[node] <= vehicle_capacity:
                if D[current, node] < min_dist:
                    min_dist = D[current, node]
                    nearest = node
        
        if nearest is None:
            # 返回仓库，开始新路径
            initial_lower_bound += D[current, depot]
            current = depot
            temp_demand = 0
            temp_path.append(depot)
        else:
            # 前往最近节点
            initial_lower_bound += min_dist
            temp_demand += demands[nearest]
            current = nearest
            temp_path.append(nearest)
            unvisited.remove(nearest)
    
    # 最后返回仓库
    initial_lower_bound += D[current, depot]
    
    # 重置未访问节点
    unvisited = set(range(1, n))
    
    # 创建根节点
    root_node = Node(
        level=0,
        path=[depot],
        cost=0,
        demand=0,
        lower_bound=initial_lower_bound
    )
    
    heapq.heappush(priority_queue, root_node)
    
    print(f"\n开始分支定界搜索...")
    print(f"初始下界: {initial_lower_bound:.2f}")
    
    # 主循环
    iteration = 0
    while priority_queue and (time.time() - start_time) < time_limit:
        iteration += 1
        
        # 获取下界最小的节点
        current_node = heapq.heappop(priority_queue)
        nodes_explored += 1
        
        # 剪枝：如果当前节点的下界已经大于已知最优解
        if current_node.lower_bound >= best_cost - 1e-9:
            pruned_nodes += 1
            continue
        
        # 如果已访问所有节点
        if len(current_node.path) == n:
            # 添加返回仓库的成本
            complete_cost = current_node.cost + D[current_node.path[-1], depot]
            
            if complete_cost < best_cost - 1e-9:
                best_cost = complete_cost
                best_solution = current_node.path[:] + [depot]
                
                print(f"迭代 {iteration}: 找到新最优解 {best_cost:.2f}")
            
            continue
        
        # 生成子节点
        last_node = current_node.path[-1]
        remaining_nodes = set(range(1, n)) - set(current_node.path)
        
        for next_node in remaining_nodes:
            # 检查容量约束
            new_demand = current_node.demand + demands[next_node]
            if new_demand > vehicle_capacity + 1e-9:
                continue
            
            # 检查是否在禁止边中
            if (last_node, next_node) in current_node.excluded_edges:
                continue
            
            # 计算新成本
            new_cost = current_node.cost + D[last_node, next_node]
            
            # 计算新路径
            new_path = current_node.path[:] + [next_node]
            
            # 计算新的下界
            remaining_after_move = remaining_nodes - {next_node}
            new_lower_bound = calculate_lower_bound(
                D, new_path, remaining_after_move, new_cost, 
                vehicle_capacity, vehicle_capacity, depot
            )
            
            # 如果下界超过当前最优解，剪枝
            if new_lower_bound >= best_cost - 1e-9:
                pruned_nodes += 1
                continue
            
            # 创建新节点
            new_node = Node(
                level=current_node.level + 1,
                path=new_path,
                cost=new_cost,
                demand=new_demand,
                lower_bound=new_lower_bound,
                excluded_edges=current_node.excluded_edges.copy()
            )
            
            heapq.heappush(priority_queue, new_node)
        
        # 添加返回仓库的子节点（完成当前车辆路径）
        if current_node.path[-1] != depot:
            return_cost = D[current_node.path[-1], depot]
            new_cost = current_node.cost + return_cost
            new_path = current_node.path[:] + [depot]
            
            # 从depot重新开始
            new_path.append(depot)
            
            new_lower_bound = calculate_lower_bound(
                D, [depot], remaining_nodes, new_cost, 
                0, vehicle_capacity, depot
            )
            
            if new_lower_bound < best_cost - 1e-9:
                new_node = Node(
                    level=current_node.level + 1,
                    path=new_path,
                    cost=new_cost,
                    demand=0,  # 新车辆，需求重置
                    lower_bound=new_lower_bound,
                    excluded_edges=current_node.excluded_edges.copy()
                )
                heapq.heappush(priority_queue, new_node)
        
        # 每1000次迭代打印进度
        if iteration % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"迭代 {iteration}: 已探索 {nodes_explored} 节点, "
                  f"剪枝 {pruned_nodes} 节点, "
                  f"当前最优 {best_cost:.2f}, "
                  f"队列大小 {len(priority_queue)}, "
                  f"时间 {elapsed:.1f}s")
    
    elapsed_time = time.time() - start_time
    
    print(f"\n搜索完成!")
    print(f"总迭代次数: {iteration}")
    print(f"探索节点数: {nodes_explored}")
    print(f"剪枝节点数: {pruned_nodes}")
    print(f"计算时间: {elapsed_time:.2f}秒")
    
    if best_solution is not None:
        print(f"最优成本: {best_cost:.2f}")
    else:
        print("未找到可行解")
    
    return best_solution, best_cost, nodes_explored

def extract_routes_from_solution(solution, depot=0):
    """从解决方案中提取各个车辆的路径"""
    if not solution:
        return []
    
    routes = []
    current_route = []
    
    for node in solution:
        if node == depot and current_route:
            routes.append([depot] + current_route + [depot])
            current_route = []
        elif node != depot:
            current_route.append(node)
    
    return routes

def calculate_route_costs(routes, D):
    """计算每条路径的成本"""
    route_costs = []
    for route in routes:
        cost = 0
        for i in range(len(route)-1):
            cost += D[route[i], route[i+1]]
        route_costs.append(cost)
    return route_costs


def generate_random_problem(num_customers=8, area_size=100, max_demand=15, vehicle_capacity=30):
    """生成随机CVRP问题"""
    # 固定随机种子以便复现
    np.random.seed(42)
    
    # 生成仓库位置（在中心）
    depot = (area_size/2, area_size/2)
    
    # 生成客户位置
    customers = []
    for i in range(num_customers):
        x = np.random.uniform(0, area_size)
        y = np.random.uniform(0, area_size)
        customers.append((x, y))
    
    # 生成需求
    demands = [0]  # 仓库需求为0
    for i in range(num_customers):
        demands.append(np.random.randint(1, max_demand+1))
    
    # 所有点（仓库+客户）
    all_points = [depot] + customers
    
    return all_points, demands, vehicle_capacity

def greedy_heuristic(D, demands, vehicle_capacity, depot=0):
    """贪心算法作为对比基准"""
    n = len(D)
    unvisited = set(range(1, n))
    routes = []
    current_route = [depot]
    current_demand = 0
    current_cost = 0
    
    while unvisited:
        # 找到最近的未访问节点
        last_node = current_route[-1]
        nearest = None
        min_dist = float('inf')
        
        for node in unvisited:
            if current_demand + demands[node] <= vehicle_capacity:
                if D[last_node, node] < min_dist:
                    min_dist = D[last_node, node]
                    nearest = node
        
        if nearest is None:
            # 返回仓库，完成当前路径
            return_cost = D[last_node, depot]
            current_cost += return_cost
            current_route.append(depot)
            routes.append((current_route, current_cost, current_demand))
            
            # 开始新路径
            current_route = [depot]
            current_demand = 0
            current_cost = 0
        else:
            # 前往最近节点
            current_cost += min_dist
            current_demand += demands[nearest]
            current_route.append(nearest)
            unvisited.remove(nearest)
    
    # 处理最后一条路径
    if len(current_route) > 1:
        return_cost = D[current_route[-1], depot]
        current_cost += return_cost
        current_route.append(depot)
        routes.append((current_route, current_cost, current_demand))
    
    # 计算总成本
    total_cost = sum(cost for _, cost, _ in routes)
    
    return routes, total_cost

def main():
    """主函数"""
    print("=" * 60)
    print("CVRP精确求解器 - Branch and Bound算法")
    print("=" * 60)
    
    # 生成小规模问题（Branch and Bound对规模敏感）
    num_customers = 8
    area_size = 100
    max_demand = 10
    vehicle_capacity = 25
    
    print(f"\n生成CVRP问题...")
    print(f"  客户数量: {num_customers} (小规模以便B&B求解)")
    print(f"  区域大小: {area_size}x{area_size}")
    print(f"  最大需求: {max_demand}")
    print(f"  车辆容量: {vehicle_capacity}")
    
    points, demands, capacity = generate_random_problem(
        num_customers, area_size, max_demand, vehicle_capacity
    )
    
    # 显示问题信息
    print(f"\n客户信息:")
    print("  客户 |  X坐标  |  Y坐标  | 需求")
    print("  " + "-" * 30)
    for i in range(1, len(points)):
        print(f"  {i:3d} | {points[i][0]:6.1f} | {points[i][1]:6.1f} | {demands[i]:3d}")
    
    print(f"\n仓库位置: ({points[0][0]:.1f}, {points[0][1]:.1f})")
    print(f"总需求: {sum(demands)}")
    print(f"最小所需车辆数: {math.ceil(sum(demands) / capacity)}")
    
    # 计算距离矩阵
    D = calculate_distance_matrix(points)
    
    # 运行贪心算法作为基准
    print(f"\n" + "="*60)
    print("贪心算法基准解:")
    greedy_routes, greedy_cost = greedy_heuristic(D, demands, capacity)
    
    print(f"贪心算法成本: {greedy_cost:.2f}")
    print(f"使用车辆: {len(greedy_routes)}")
    for i, (route, cost, demand) in enumerate(greedy_routes):
        print(f"  车辆{i+1}: {route} (距离: {cost:.2f}, 需求: {demand})")
    
    # 运行Branch and Bound算法
    print(f"\n" + "="*60)
    time_limit = 60  
    
    best_solution, best_cost, nodes_explored = branch_and_bound_cvrp(
        D, demands, capacity, time_limit
    )
    
    if best_solution is not None:
        # 提取路径
        routes = extract_routes_from_solution(best_solution)
        route_costs = calculate_route_costs(routes, D)
        
        print(f"\n最优解决方案:")
        print(f"总成本: {best_cost:.2f}")
        print(f"使用车辆: {len(routes)}")
        
        total_demand_covered = 0
        for i, route in enumerate(routes):
            route_demand = sum(demands[node] for node in route[1:-1])
            total_demand_covered += route_demand
            print(f"  车辆{i+1}: {route} (距离: {route_costs[i]:.2f}, 需求: {route_demand})")
        
        print(f"覆盖总需求: {total_demand_covered}/{sum(demands)}")
        
        # 与贪心算法比较
        if best_cost < float('inf'):
            improvement = ((greedy_cost - best_cost) / greedy_cost * 100) if greedy_cost > 0 else 0
            print(f"\n与贪心算法比较:")
            print(f"  贪心算法成本: {greedy_cost:.2f}")
            print(f"  B&B最优成本: {best_cost:.2f}")
            print(f"  改进: {improvement:.1f}%")
      
    else:
        print("未找到可行解")
    
    print(f"\n" + "="*60)
    print("求解完成!")
    print("=" * 60)
    
    return best_solution, best_cost

if __name__ == "__main__":
    best_solution, best_cost = main()