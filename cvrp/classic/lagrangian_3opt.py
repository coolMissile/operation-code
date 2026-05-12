#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""拉格朗日松弛3-opt*算法
参考: Stewart & Golden (1984) Lagrangian relaxed 3-opt* heuristic
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
            dx = points[i][0] - points[j][0]
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

def solution_to_routes(solution):
    """将[0,1,2,0,3,4,0]转换为[[0,1,2,0], [0,3,4,0]]格式"""
    routes = []
    current_route = [0]
    
    for node in solution[1:]:  # 跳过第一个0
        if node == 0 and current_route:
            current_route.append(0)
            if len(current_route) > 2:  # 不只是[0,0]
                routes.append(current_route)
            current_route = [0]
        elif node != 0:
            current_route.append(node)
    
    return routes

def routes_to_solution(routes):
    """将[[0,1,2,0], [0,3,4,0]]转换为[0,1,2,0,3,4,0]格式"""
    solution = []
    for route in routes:
        solution.extend(route[:-1])  # 不包含最后一个0
    solution.append(0)  # 添加最后的0
    return solution

def remove_empty_routes(solution):
    """移除空路径[0,0]"""
    routes = solution_to_routes(solution)
    return routes_to_solution(routes)

def is_solution_feasible(solution, demands, capacity, D, max_length=None):
    """检查解是否满足约束"""
    routes = solution_to_routes(solution)
    
    for route in routes:
        # 检查容量约束
        route_demand = sum(demands[node] for node in route[1:-1])
        if route_demand > capacity + 1e-9:
            return False
        
        # 检查最大长度约束
        if max_length is not None:
            route_length = calculate_route_length(route, D)
            if route_length > max_length + 1e-9:
                return False
    
    return True

def calculate_penalized_cost(solution, D, demands, capacity, max_length, lambda_C, lambda_L):
    """
    计算惩罚后的总成本
    总成本 = 总距离 + λ_C×容量惩罚 + λ_L×长度惩罚
    """
    routes = solution_to_routes(solution)
    
    total_distance = 0
    total_capacity_penalty = 0
    total_length_penalty = 0
    
    for route in routes:
        # 计算路径距离
        route_distance = calculate_route_length(route, D)
        total_distance += route_distance
        
        # 计算容量惩罚
        route_demand = sum(demands[node] for node in route[1:-1])
        if route_demand > capacity:
            capacity_violation = route_demand - capacity
            total_capacity_penalty += lambda_C * capacity_violation
        
        # 计算长度惩罚
        if max_length is not None and route_distance > max_length:
            length_violation = route_distance - max_length
            total_length_penalty += lambda_L * length_violation
    
    total_cost = total_distance + total_capacity_penalty + total_length_penalty
    
    return total_cost, total_distance, total_capacity_penalty, total_length_penalty

def generate_initial_solution_tsp(D, customers):
    """用TSP最近邻生成初始解"""
    if not customers:
        return [0, 0]
    
    # 生成TSP路径
    unvisited = set(customers)
    current = customers[0]
    tsp_path = [current]
    unvisited.remove(current)
    
    while unvisited:
        next_node = min(unvisited, key=lambda x: D[current, x])
        tsp_path.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    
    # 转换为VRP解：整个TSP路径作为一条路径
    solution = [0] + tsp_path + [0]
    return solution

def generate_initial_solution_random(customers):
    """随机生成初始解"""
    if not customers:
        return [0, 0]
    
    shuffled = customers.copy()
    random.shuffle(shuffled)
    solution = [0] + shuffled + [0]
    return solution

def get_all_3opt_star_moves(routes):
    """生成所有可能的3-opt*移动
    
    3-opt*移动涉及3条边，可以在同一条路径内，也可以跨路径
    """
    moves = []
    
    # 将路径展平为节点序列
    nodes = []
    route_breaks = []  # 记录每条路径结束的位置
    
    for route in routes:
        # 去掉首尾的0（除了第一条路径的开头）
        route_nodes = route[1:-1] if len(route) > 2 else []
        nodes.extend(route_nodes)
        if route_nodes:
            route_breaks.append(len(nodes))
    
    n = len(nodes)
    if n < 3:
        return moves
    
    # 生成所有可能的3条边组合
    # 简化版本：只考虑相邻的三条边
    for i in range(n-2):
        for j in range(i+1, n-1):
            for k in range(j+1, n):
                # 检查这3个点是否在不同路径中
                # 这决定了是intra-route还是inter-route移动
                moves.append((i, j, k))
    
    return moves[:50]  # 限制移动数量，防止组合爆炸

def apply_3opt_star_move(solution, move_type, i, j, k):
    """应用3-opt*移动
    
    移动类型：
    0: 标准3-opt (intra-route)
    1: 交换两条路径的段 (inter-route)
    2: 移动段到另一条路径
    """
    routes = solution_to_routes(solution)
    
    if move_type == 0:
        # 标准3-opt: 反转段
        # 这里简化处理，实际3-opt有更多可能
        pass
    elif move_type == 1:
        # 交换两条路径的段
        # 这里简化处理
        pass
    
    return routes_to_solution(routes)

def find_best_3opt_star_move(solution, D, demands, capacity, max_length, 
                            lambda_C, lambda_L, max_moves=1000):
    """寻找最佳的3-opt*移动"""
    routes = solution_to_routes(solution)
    
    best_delta = float('inf')
    best_move = None
    best_new_solution = None
    
    # 当前惩罚成本
    current_cost, _, _, _ = calculate_penalized_cost(
        solution, D, demands, capacity, max_length, lambda_C, lambda_L
    )
    
    # 尝试的移动计数器
    move_count = 0
    
    # 简化：尝试随机移动
    for _ in range(min(max_moves, 100)):
        if len(routes) < 2:
            break
            
        # 随机选择两条路径
        r1_idx, r2_idx = random.sample(range(len(routes)), 2)
        r1 = routes[r1_idx]
        r2 = routes[r2_idx]
        
        # 如果路径太短，跳过
        if len(r1) <= 3 or len(r2) <= 3:
            continue
        
        # 尝试交换一个客户
        # 从r1中间选一个客户（不是首尾的0）
        c1_idx = random.randint(1, len(r1)-2)
        customer1 = r1[c1_idx]
        
        # 从r2中间选一个插入位置
        c2_idx = random.randint(1, len(r2)-1)  # 可以插入到末尾之前
        
        # 创建新解
        new_r1 = r1[:c1_idx] + r1[c1_idx+1:]  # 移除customer1
        new_r2 = r2[:c2_idx] + [customer1] + r2[c2_idx:]  # 插入customer1
        
        # 更新路径列表
        new_routes = routes.copy()
        new_routes[r1_idx] = new_r1
        new_routes[r2_idx] = new_r2
        
        # 转换为解格式
        new_solution = routes_to_solution(new_routes)
        
        # 计算新成本
        new_cost, _, _, _ = calculate_penalized_cost(
            new_solution, D, demands, capacity, max_length, lambda_C, lambda_L
        )
        
        delta = new_cost - current_cost
        move_count += 1
        
        if delta < best_delta - 1e-9:
            best_delta = delta
            best_move = (r1_idx, c1_idx, r2_idx, c2_idx)
            best_new_solution = new_solution
        
        # 如果找到了改善的移动，可以提前返回
        if best_delta < -1e-9 and move_count > 10:
            break
    
    return best_delta, best_move, best_new_solution

def force_solution_feasible(solution, demands, capacity, D, max_length=None):
    """强制使解可行（通过插入仓库分割超载路径）"""
    routes = solution_to_routes(solution)
    feasible_routes = []
    
    for route in routes:
        if len(route) <= 2:  # 空路径或只有[0,0]
            continue
            
        current_route = [0]
        current_demand = 0
        current_length = 0
        prev_node = 0
        
        for node in route[1:]:  # 从第一个客户开始
            if node == 0:
                continue
                
            # 检查添加这个节点是否会导致违反约束
            new_demand = current_demand + demands[node]
            new_length = current_length + D[prev_node, node] + D[node, 0]
            
            capacity_ok = new_demand <= capacity
            length_ok = (max_length is None) or (new_length <= max_length)
            
            if not (capacity_ok and length_ok) and len(current_route) > 1:
                # 结束当前路径，开始新路径
                current_route.append(0)
                feasible_routes.append(current_route)
                
                # 开始新路径
                current_route = [0, node]
                current_demand = demands[node]
                current_length = D[0, node] + D[node, 0]
            else:
                # 添加到当前路径
                current_route.append(node)
                current_demand = new_demand
                current_length = new_length - D[node, 0]  # 去掉回仓库的距离
            
            prev_node = node
        
        # 添加最后一条路径
        if len(current_route) > 1:
            current_route.append(0)
            feasible_routes.append(current_route)
    
    return routes_to_solution(feasible_routes)

def lagrangian_3opt_star(points, demands, capacity, D,
                         max_length=None,
                         initial_lambda_C=0.1,
                         initial_lambda_L=0.1,
                         initialization="tsp",
                         max_iterations=100,
                         lambda_multiplier=2.0,
                         verbose=True):
    """
    拉格朗日松弛3-opt*算法主函数
    
    Args:
        initialization: 初始化解方法，"tsp"或"random"
        max_iterations: 最大迭代次数
        lambda_multiplier: λ的倍增因子
        verbose: 是否打印详细信息
    """
    n = len(points)
    customers = list(range(1, n))
    
    print("=" * 70)
    print("拉格朗日松弛3-opt*算法 (Stewart & Golden, 1984)")
    print("=" * 70)
    print(f"参数: 初始化={initialization}, λ_C={initial_lambda_C}, "
          f"λ_L={initial_lambda_L}, 最大迭代={max_iterations}")
    if max_length:
        print(f"最大路径长度约束: {max_length}")
    
    start_time = time.time()
    
    # 阶段1: 生成初始解
    print(f"\n阶段1: 生成初始解...")
    
    if initialization == "tsp":
        solution = generate_initial_solution_tsp(D, customers)
    elif initialization == "random":
        solution = generate_initial_solution_random(customers)
    else:
        raise ValueError(f"未知的初始化方法: {initialization}")
    
    # 初始λ值
    lambda_C = initial_lambda_C
    lambda_L = initial_lambda_L if max_length is not None else 0
    
    # 记录历史
    history = {
        'iterations': [],
        'total_distance': [],
        'total_cost': [],
        'capacity_penalty': [],
        'length_penalty': [],
        'lambda_C': [],
        'lambda_L': [],
        'feasible': [],
        'num_routes': []
    }
    
    # 主循环
    print(f"\n阶段2: 拉格朗日松弛优化...")
    
    iteration = 0
    last_improvement = 0
    
    while iteration < max_iterations:
        iteration += 1
        if verbose and iteration % 10 == 0:
            print(f"  迭代 {iteration}, λ_C={lambda_C:.4f}, λ_L={lambda_L:.4f}")
        
        # 计算当前成本
        current_cost, current_distance, capacity_penalty, length_penalty = calculate_penalized_cost(
            solution, D, demands, capacity, max_length, lambda_C, lambda_L
        )
        
        # 记录历史
        history['iterations'].append(iteration)
        history['total_distance'].append(current_distance)
        history['total_cost'].append(current_cost)
        history['capacity_penalty'].append(capacity_penalty)
        history['length_penalty'].append(length_penalty)
        history['lambda_C'].append(lambda_C)
        history['lambda_L'].append(lambda_L)
        
        # 检查可行性
        feasible = is_solution_feasible(solution, demands, capacity, D, max_length)
        history['feasible'].append(feasible)
        
        routes = solution_to_routes(solution)
        history['num_routes'].append(len(routes))
        
        # 如果解可行且已经稳定，可以提前终止
        if feasible and iteration - last_improvement > 5:
            if verbose:
                print(f"  找到可行解且已稳定，提前终止")
            break
        
        # 寻找最佳3-opt*移动
        best_delta, best_move, best_new_solution = find_best_3opt_star_move(
            solution, D, demands, capacity, max_length, lambda_C, lambda_L
        )
        
        if best_delta < -1e-9:  # 找到了改善的移动
            solution = best_new_solution
            last_improvement = iteration
            
            if verbose and iteration % 5 == 0:
                routes = solution_to_routes(solution)
                print(f"  迭代 {iteration}: 距离={current_distance:.2f}, "
                      f"惩罚={capacity_penalty+length_penalty:.2f}, "
                      f"车辆数={len(routes)}")
        else:
            # 没有改善，增加λ值
            lambda_C *= lambda_multiplier
            if max_length is not None:
                lambda_L *= lambda_multiplier
            
            # 如果λ太大，强制使解可行
            if lambda_C > 1e6 or (max_length is not None and lambda_L > 1e6):
                if verbose:
                    print(f"  λ值过大，强制使解可行")
                solution = force_solution_feasible(solution, demands, capacity, D, max_length)
                break
    
    # 最终修复（确保解可行）
    if not is_solution_feasible(solution, demands, capacity, D, max_length):
        if verbose:
            print(f"  最终修复不可行解")
        solution = force_solution_feasible(solution, demands, capacity, D, max_length)
    
    # 最终统计
    routes = solution_to_routes(solution)
    total_distance = calculate_total_distance(routes, D)
    feasible = is_solution_feasible(solution, demands, capacity, D, max_length)
    
    elapsed_time = time.time() - start_time
    
    # 输出结果
    print(f"\n" + "=" * 70)
    print("算法完成！")
    print(f"总迭代次数: {iteration}")
    print(f"总路径数: {len(routes)}")
    print(f"总距离: {total_distance:.2f}")
    print(f"可行性: {'✓' if feasible else '✗'}")
    print(f"运行时间: {elapsed_time:.3f} 秒")
    
    # 输出路径详情
    print(f"\n详细路径:")
    for i, route in enumerate(routes):
        route_demand = sum(demands[node] for node in route[1:-1])
        route_length = calculate_route_length(route, D)
        utilization = route_demand / capacity * 100 if capacity > 0 else 0
        
        constraints_violated = []
        if route_demand > capacity + 1e-9:
            constraints_violated.append(f"容量超载{route_demand-capacity:.1f}")
        if max_length and route_length > max_length + 1e-9:
            constraints_violated.append(f"长度超限{route_length-max_length:.1f}")
        
        constraint_status = " ✓" if not constraints_violated else f" ✗({', '.join(constraints_violated)})"
        
        print(f"  路径{i+1}: {route} (需求: {route_demand}/{capacity}, "
              f"距离: {route_length:.2f}, 利用率: {utilization:.1f}%){constraint_status}")
    
    # 算法统计
    print(f"\n算法统计:")
    print(f"  最终λ_C: {lambda_C:.4f}")
    print(f"  最终λ_L: {lambda_L:.4f}")
    print(f"  历史最佳距离: {min(history['total_distance']):.2f}")
    print(f"  可行解迭代: {sum(history['feasible'])}/{len(history['feasible'])}")
    
    return routes, total_distance, history

def generate_random_problem(num_customers=20, area_size=100, max_demand=20, vehicle_capacity=50):
    """生成随机VRP问题"""
    # 仓库位置在中心
    depot = (area_size/2, area_size/2)
    
    # 生成客户位置
    customers = []
    for i in range(num_customers):
        x = random.uniform(0, area_size)
        y = random.uniform(0, area_size)
        customers.append((x, y))
    
    # 生成需求
    demands = [0]  # 仓库需求为0
    for i in range(num_customers):
        demands.append(random.randint(1, max_demand))
    
    # 所有点
    all_points = [depot] + customers
    
    return all_points, demands, vehicle_capacity

def compare_lambda_strategies(points, demands, capacity, D, max_length=None):
    """比较不同的λ策略"""
    print("=" * 70)
    print("不同λ策略比较")
    print("=" * 70)
    
    strategies = [
        ("温和增长", 0.05, 0.05, 1.5),
        ("快速增长", 0.1, 0.1, 2.0),
        ("激进增长", 0.2, 0.2, 3.0),
    ]
    
    results = {}
    
    for name, lambda_C, lambda_L, multiplier in strategies:
        print(f"\n策略: {name} (λ_C={lambda_C}, λ_L={lambda_L}, 乘子={multiplier})")
        
        try:
            routes, distance, history = lagrangian_3opt_star(
                points, demands, capacity, D,
                max_length=max_length,
                initial_lambda_C=lambda_C,
                initial_lambda_L=lambda_L,
                initialization="tsp",
                max_iterations=50,
                lambda_multiplier=multiplier,
                verbose=False
            )
            
            results[name] = {
                'distance': distance,
                'routes': routes,
                'history': history,
                'num_routes': len(routes)
            }
            
            print(f"  结果: 距离={distance:.2f}, 车辆数={len(routes)}")
            
        except Exception as e:
            print(f"  失败: {e}")
    
    return results

def main():
    """主函数"""
    # 参数设置
    num_customers = 15
    area_size = 100
    max_demand = 15
    vehicle_capacity = 40
    
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
    
    # 设置最大路径长度（可选）
    max_length = None
    # max_length = 150  # 可以取消注释启用长度约束
    
    # 比较不同的λ策略
    comparison_results = compare_lambda_strategies(points, demands, capacity, D, max_length)
    
    # 选择最佳策略进行详细运行
    if comparison_results:
        best_strategy = min(comparison_results.items(), 
                           key=lambda x: x[1]['distance'])
        
        print(f"\n" + "="*70)
        print(f"最佳策略: {best_strategy[0]}")
        print("=" * 70)
        
        # 使用最佳参数重新运行（详细模式）
        if best_strategy[0] == "温和增长":
            lambda_C, lambda_L, multiplier = 0.05, 0.05, 1.5
        elif best_strategy[0] == "快速增长":
            lambda_C, lambda_L, multiplier = 0.1, 0.1, 2.0
        else:
            lambda_C, lambda_L, multiplier = 0.2, 0.2, 3.0
        
        final_routes, final_distance, history = lagrangian_3opt_star(
            points, demands, capacity, D,
            max_length=max_length,
            initial_lambda_C=lambda_C,
            initial_lambda_L=lambda_L,
            initialization="tsp",
            max_iterations=100,
            lambda_multiplier=multiplier,
            verbose=True
        )
        
       
    
    print(f"\n" + "="*60)
    print("求解完成！")
    print("=" * 60)
    
    return {
        'points': points,
        'demands': demands,
        'capacity': capacity,
        'comparison_results': comparison_results,
        'final_routes': final_routes if 'final_routes' in locals() else None,
        'final_distance': final_distance if 'final_distance' in locals() else None
    }

if __name__ == "__main__":
    result = main()