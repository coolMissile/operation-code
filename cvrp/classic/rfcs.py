#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""先定线后分组算法 (Route-First Cluster-Second)
参考: Beasley (1983) 的RFCS启发式,由https://github.com/yorak/VeRyPy简化
"""

import numpy as np
import random
import time


def calculate_distance_matrix(points):
    """计算欧氏距离矩阵"""
    n = len(points)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dx = points[i][0] - points[j][1]
            dy = points[i][1] - points[j][1]
            D[i, j] = np.sqrt(dx*dx + dy*dy)
    return D

def calculate_route_length(route, D):
    """计算单条路径的长度"""
    length = 0
    for k in range(len(route) - 1):
        length += D[route[k], route[k+1]]
    return length

def calculate_total_distance(routes, D):
    """计算所有路径的总长度"""
    total = 0
    for route in routes:
        total += calculate_route_length(route, D)
    return total

def solve_tsp_nearest_neighbor(D, nodes):
    """
    最近邻算法求解TSP（简化版，替代LKH）
    
    Args:
        D: 距离矩阵
        nodes: 要访问的节点列表
    Returns:
        tour: TSP路径
        length: 路径长度
    """
    if not nodes:
        return [], 0
    unvisited = set(nodes)
    current = nodes[0]
    tour = [current]
    unvisited.remove(current)
    
    # 最近邻贪心
    while unvisited:
        # 找到最近的未访问节点
        next_node = min(unvisited, key=lambda x: D[current, x])
        tour.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    
    # 计算回路长度（回到起点）
    length = 0
    for i in range(len(tour) - 1):
        length += D[tour[i], tour[i+1]]
    if len(tour) > 1:
        length += D[tour[-1], tour[0]]  # 回到起点
    
    return tour, length

def solve_tsp_2opt(tour, D):
    """
    2-opt局部搜索优化TSP路径
    
    Args:
        tour: 初始TSP路径
        D: 距离矩阵
    Returns:
        optimized_tour: 优化后的路径
        length: 优化后的长度
    """
    if len(tour) <= 3:
        return tour, calculate_route_length(tour, D) if len(tour) > 1 else 0
    
    improved = True
    best_tour = tour.copy()
    best_length = calculate_route_length(best_tour + [best_tour[0]], D) if len(best_tour) > 1 else 0
    
    while improved:
        improved = False
        for i in range(len(best_tour) - 2):
            for j in range(i + 2, len(best_tour)):
                if j - i == 1:  
                    continue
                
                # 2-opt交换
                new_tour = best_tour.copy()
                new_tour[i+1:j+1] = reversed(new_tour[i+1:j+1])
                
                new_length = calculate_route_length(new_tour + [new_tour[0]], D) if len(new_tour) > 1 else 0
                
                if new_length < best_length - 1e-9:
                    best_tour = new_tour
                    best_length = new_length
                    improved = True
                    break
            if improved:
                break
    
    return best_tour, best_length

def optimal_partition(giant_tour, demands, capacity, D, max_route_length=None):
    """
    将巨型环路最优分割为车辆路径（动态规划）
    
    Args:
        giant_tour: 巨型TSP环路（不含仓库）
        demands: 客户需求列表
        capacity: 车辆容量
        D: 距离矩阵
        max_route_length: 最大路径长度约束
    Returns:
        routes: 分割后的路径列表
        total_distance: 总距离
    """
    n = len(giant_tour)
    
    # 如果TSP路径为空
    if n == 0:
        return [], 0
    
    # 预处理：计算所有子路径的成本
    # cost[i][j]: 从giant_tour[i]到giant_tour[j]（包括）作为一条路径的成本
    # demand[i][j]: 对应的总需求
    # feasible[i][j]: 是否满足约束
    
    cost = np.full((n, n), np.inf)
    total_demand = np.zeros((n, n), dtype=int)
    feasible = np.zeros((n, n), dtype=bool)
    
    for i in range(n):
        cumulative_demand = 0
        current_route = [0]  # 从仓库开始
        
        for j in range(i, n):
            # 添加客户j
            customer = giant_tour[j]
            cumulative_demand += demands[customer]
            current_route.append(customer)
            
            # 检查容量约束
            if cumulative_demand > capacity:
                break
            
            # 计算路径成本（包括回到仓库）
            test_route = current_route + [0]
            route_length = calculate_route_length(test_route, D)
            
            # 检查最大长度约束
            if max_route_length is not None and route_length > max_route_length:
                break
            
            # 记录
            cost[i, j] = route_length
            total_demand[i, j] = cumulative_demand
            feasible[i, j] = True
    
    # 动态规划：找到最优分割
    # dp[i]: 分割前i个客户的最小成本
    # prev[i]: 前一个分割点
    
    dp = np.full(n + 1, np.inf)
    dp[0] = 0
    prev = np.full(n + 1, -1, dtype=int)
    
    for j in range(1, n + 1):
        for i in range(j):
            if feasible[i, j-1] and dp[i] + cost[i, j-1] < dp[j]:
                dp[j] = dp[i] + cost[i, j-1]
                prev[j] = i
    
    # 如果无法找到可行解（理论上不会发生，但以防万一）
    if np.isinf(dp[n]):
        # 使用贪心分割作为备用方案
        return greedy_partition(giant_tour, demands, capacity, D, max_route_length)
    
    # 回溯重建路径
    routes = []
    j = n
    while j > 0:
        i = prev[j]
        if i < 0:  # 没有前驱，出错
            break
        
        # 获取这条路径
        route = [0] + giant_tour[i:j] + [0]
        routes.append(route)
        j = i
    
    # 反转顺序（因为是从后往前回溯的）
    routes.reverse()
    
    # 计算总距离
    total_distance = dp[n]
    
    return routes, total_distance

def greedy_partition(giant_tour, demands, capacity, D, max_route_length=None):
    """
    贪心分割作为备用方案
    """
    routes = []
    current_route = [0]
    current_demand = 0
    current_length = 0
    
    for customer in giant_tour:
        # 检查添加这个客户是否可行
        new_demand = current_demand + demands[customer]
        
        # 临时路径计算长度
        temp_route = current_route + [customer, 0]
        temp_length = calculate_route_length(temp_route, D)
        
        # 检查约束
        capacity_ok = new_demand <= capacity
        length_ok = (max_route_length is None) or (temp_length <= max_route_length)
        
        if capacity_ok and length_ok:
            # 可以添加到当前路径
            current_route.append(customer)
            current_demand = new_demand
            current_length = temp_length
        else:
            # 结束当前路径，开始新路径
            if len(current_route) > 1:  # 不只包含仓库
                current_route.append(0)
                routes.append(current_route)
            
            # 开始新路径
            current_route = [0, customer]
            current_demand = demands[customer]
            current_length = calculate_route_length(current_route + [0], D)
    
    # 添加最后一条路径
    if len(current_route) > 1:
        current_route.append(0)
        routes.append(current_route)
    
    # 计算总距离
    total_distance = calculate_total_distance(routes, D)
    
    return routes, total_distance

def route_first_cluster_second(points, demands, capacity, D, 
                              tsp_solver="nearest_neighbor",
                              use_2opt=True,
                              max_route_length=None):
    """
    先定线后分组算法主函数
    
    Args:
        tsp_solver: TSP求解器，可选 "nearest_neighbor" 或 "random"
        use_2opt: 是否使用2-opt优化TSP路径
        max_route_length: 最大路径长度约束
    """
    n = len(points)
    
    print(f"参数: TSP求解器={tsp_solver}, 2-opt优化={use_2opt}")
    if max_route_length:
        print(f"最大路径长度约束: {max_route_length}")
    
    start_time = time.time()
    
    # 阶段1: 生成巨型TSP环路（不包括仓库）
    print(f"\n阶段1: 生成巨型TSP环路...")
    print(f"  客户数量: {n-1} (不包括仓库)")
    
    customers = list(range(1, n))
    
    if tsp_solver == "nearest_neighbor":
        # 使用最近邻算法生成初始TSP解
        giant_tour, tsp_length = solve_tsp_nearest_neighbor(D, customers)
    elif tsp_solver == "random":
        # 随机TSP路径
        giant_tour = customers.copy()
        random.shuffle(giant_tour)
        tsp_length = calculate_route_length(giant_tour + [giant_tour[0]], D) if giant_tour else 0
    else:
        raise ValueError(f"未知的TSP求解器: {tsp_solver}")
    
    print(f"  初始TSP路径长度: {tsp_length:.2f}")
    print(f"  TSP路径: {giant_tour}")
    
    # 可选：使用2-opt优化TSP路径
    if use_2opt and len(giant_tour) > 3:
        print(f"  使用2-opt优化TSP路径...")
        giant_tour, tsp_length = solve_tsp_2opt(giant_tour, D)
        print(f"  优化后TSP长度: {tsp_length:.2f}")
        print(f"  优化后TSP路径: {giant_tour}")
    
    # 阶段2: 最优分割
    print(f"\n阶段2: 最优分割为车辆路径...")
    
    routes, total_distance = optimal_partition(
        giant_tour, demands, capacity, D, max_route_length
    )
    
    elapsed_time = time.time() - start_time
   
    print(f"\n" + "=" * 70)
    print("算法完成！")
    print(f"总路径数: {len(routes)}")
    print(f"总距离: {total_distance:.2f}")
    print(f"运行时间: {elapsed_time:.3f} 秒")
    
    # 与直接服务每个客户的对比
    direct_distance = 0
    for i in range(1, n):
        direct_distance += 2 * D[0, i]  # 仓库->客户->仓库
    
    print(f"\n对比分析:")
    print(f"  每客户单独服务总距离: {direct_distance:.2f}")
    print(f"  TSP环路长度: {tsp_length:.2f}")
    print(f"  RFCS算法总距离: {total_distance:.2f}")
    print(f"  相比于单独服务节约: {(direct_distance - total_distance)/direct_distance*100:.2f}%")
    
    print(f"\n详细路径:")
    for i, route in enumerate(routes):
        route_demand = sum(demands[node] for node in route[1:-1])
        route_length = calculate_route_length(route, D)
        utilization = route_demand / capacity * 100
        print(f"  路径{i+1}: {route} (需求: {route_demand}/{capacity}, "
              f"距离: {route_length:.2f}, 利用率: {utilization:.1f}%)")
    
    return routes, total_distance, {
        'giant_tour': giant_tour,
        'tsp_length': tsp_length,
        'direct_distance': direct_distance
    }

def generate_random_problem(num_customers=20, area_size=100, max_demand=20, vehicle_capacity=80):
    """生成随机VRP问题"""
    # 仓库位置在中心
    depot = (area_size/2, area_size/2)
    
    # 客户位置
    customers = []
    for i in range(num_customers):
        x = random.uniform(0, area_size)
        y = random.uniform(0, area_size)
        customers.append((x, y))
    
    # 需求
    demands = [0]  # 仓库需求为0
    for i in range(num_customers):
        demands.append(random.randint(1, max_demand))
    
    all_points = [depot] + customers
    
    return all_points, demands, vehicle_capacity

# def compare_with_other_algorithms(points, demands, capacity, D):
#     """与简单算法对比"""
#     n = len(points)
    
#     print(f"\n" + "="*60)
#     print("与简单算法对比:")
#     print("=" * 60)
    
#     # 1. 每客户单独服务
#     direct_distance = 0
#     for i in range(1, n):
#         direct_distance += 2 * D[0, i]
#     print(f"1. 每客户单独服务:")
#     print(f"   车辆数: {n-1}")
#     print(f"   总距离: {direct_distance:.2f}")
    
#     # 2. 简单最近邻
#     print(f"\n2. 简单最近邻算法:")
#     unvisited = set(range(1, n))
#     routes = []
#     current_route = [0]
#     current_demand = 0
    
#     while unvisited:
#         if not current_route:  # 新路径
#             current_route = [0]
#             current_demand = 0
        
#         # 找最近的未访问客户
#         last_node = current_route[-1]
#         nearest = None
#         min_dist = float('inf')
        
#         for customer in unvisited:
#             if current_demand + demands[customer] <= capacity:
#                 dist = D[last_node, customer]
#                 if dist < min_dist:
#                     min_dist = dist
#                     nearest = customer
        
#         if nearest is None:  
#             current_route.append(0)
#             routes.append(current_route)
#             current_route = [0]
#             current_demand = 0
#         else:
#             current_route.append(nearest)
#             current_demand += demands[nearest]
#             unvisited.remove(nearest)
    
#     # 添加最后一条路径
#     if len(current_route) > 1:
#         current_route.append(0)
#         routes.append(current_route)
    
#     nn_distance = calculate_total_distance(routes, D)
#     print(f"   车辆数: {len(routes)}")
#     print(f"   总距离: {nn_distance:.2f}")
    
#     return direct_distance, nn_distance, routes

def main():
    """主函数"""
    # 参数设置
    num_customers = 15  # 客户数少一点，便于观察
    area_size = 100
    max_demand = 20
    vehicle_capacity = 50  # 容量小一点，让分割更有意义
    
    print(f"\n生成随机CVRP问题...")
    print(f"  客户数量: {num_customers}")
    print(f"  区域大小: {area_size}x{area_size}")
    print(f"  最大需求: {max_demand}")
    print(f"  车辆容量: {vehicle_capacity}")
    
    # 生成问题实例
    points, demands, capacity = generate_random_problem(
        num_customers, area_size, max_demand, vehicle_capacity
    )
    
    # 显示基本信息
    print(f"\n客户信息:")
    print("  客户 |  X坐标  |  Y坐标  | 需求")
    print("  " + "-" * 30)
    for i in range(1, min(11, len(points))):
        print(f"  {i:3d} | {points[i][0]:6.1f} | {points[i][1]:6.1f} | {demands[i]:3d}")
    if len(points) > 11:
        print(f"  ... 共{len(points)-1}个客户")
    
    print(f"\n仓库位置: ({points[0][0]:.1f}, {points[0][1]:.1f})")
    print(f"总需求: {sum(demands)}")
    print(f"理论最小车辆数: {sum(demands)/capacity:.1f}")
    
    # 计算距离矩阵
    D = calculate_distance_matrix(points)
    
    # 与简单算法对比
    # direct_dist, nn_dist, nn_routes = compare_with_other_algorithms(points, demands, capacity, D)
    
    # 运行RFCS算法
    print(f"\n" + "="*60)
    print("运行先定线后分组算法...")
    
    # 测试不同的TSP求解器
    results = {}
    
    for tsp_solver in ["nearest_neighbor", "random"]:
        print(f"\n使用TSP求解器: {tsp_solver}")
        try:
            routes, distance, info = route_first_cluster_second(
                points, demands, capacity, D,
                tsp_solver=tsp_solver,
                use_2opt=True,
                max_route_length=None
            )
            results[tsp_solver] = {
                'distance': distance,
                'routes': routes,
                'info': info
            }
        except Exception as e:
            print(f"  求解失败: {e}")
    
    # 结果对比
    print(f"\n" + "="*60)
    print("最终结果对比:")
    print("=" * 60)
    
    # print(f"1. 每客户单独服务:")
    # print(f"   距离: {direct_dist:.2f}")
    
    # print(f"\n2. 简单最近邻算法:")
    # print(f"   距离: {nn_dist:.2f}")
    # print(f"   车辆数: {len(nn_routes)}")
    
    for tsp_solver, result in results.items():
        print(f"\n3. RFCS算法 ({tsp_solver}):")
        print(f"   距离: {result['distance']:.2f}")
        print(f"   车辆数: {len(result['routes'])}")
        
        # # 与最近邻对比
        # if result['distance'] < nn_dist:
        #     improvement = (nn_dist - result['distance']) / nn_dist * 100
        #     print(f"   相比最近邻改善: {improvement:.2f}%")
    
    # 可视化最佳解
    best_result = min(results.values(), key=lambda x: x['distance'])
    best_routes = best_result['routes']
    best_distance = best_result['distance']
    
    print(f"\n" + "="*60)
    print(f"最佳解 (距离: {best_distance:.2f}):")
    for i, route in enumerate(best_routes):
        route_demand = sum(demands[node] for node in route[1:-1])
        route_length = calculate_route_length(route, D)
        print(f"  路径{i+1}: {route} (需求: {route_demand}/{capacity}, 距离: {route_length:.2f})")
    
    print(f"\n" + "="*60)
    print("求解完成！")
    print("=" * 60)
    
    return {
        'points': points,
        'demands': demands,
        'capacity': capacity,
        # 'direct_distance': direct_dist,
        # 'nn_distance': nn_dist,
        # 'nn_routes': nn_routes,
        'rfcs_results': results,
        'best_routes': best_routes,
        'best_distance': best_distance
    }

if __name__ == "__main__":
    result = main()