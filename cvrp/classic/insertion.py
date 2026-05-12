#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""插入启发式算法 (Insertion Heuristics) 简化版
参考: Mole and Jameson (1976) 的插入算法 https://github.com/yorak/VeRyPy/tree/master简化版
分为顺序和并行两种
适合辐射状不均匀分布，优先处理远距离客户，对求最小距离效果不如最近邻
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

class InsertionCandidate:
    """插入候选数据结构"""
    def __init__(self, customer, after_node, before_node, cost_delta, demand_delta):
        self.customer = customer
        self.after_node = after_node
        self.before_node = before_node
        self.cost_delta = cost_delta
        self.demand_delta = demand_delta
        self.id = id(self)  # 唯一标识符
        
    def __repr__(self):
        return f"Insert({self.customer} between {self.after_node} and {self.before_node}, Δ={self.cost_delta:.2f})"

def parametrized_insertion_criteria(D, i, u, j, lm=1.0, mm=0.0):
    """Mole and Jameson (1976) 插入准则
    
    i = 插入在这个节点之后
    u = 要插入的客户
    j = 插入在这个节点之前
    
    lm = lambda 乘子 (>=0) 控制对单客户路线的偏好
    mm = mu 乘子 (>=0) 类似于节约算法的 lambda
    
    Returns: (strain, secondary_criterion)
    """
    # 计算插入成本
    insertion_cost = D[i, u] + D[u, j] - mm * D[i, j]
    # 计算节约（插入相比于单独服务的节约）
    strain = lm * D[0, u] - insertion_cost
    return strain, insertion_cost

def sequential_insertion_heuristic(points, demands, capacity, D, 
                                  lambda_mul=2.0, mu_mul=1.0,
                                  initialize_with="closest"):
    """
    顺序插入启发式算法
    
    参数:
        lambda_mul: lambda 乘子，控制对单客户路线的偏好
        mu_mul: mu 乘子，控制插入成本的计算
        initialize_with: 初始化方式，可选 "farthest"(最远), "closest"(最近)
    """
    n = len(points)
    start_time = time.time()
    
    # 未分配客户集合
    unrouted = set(range(1, n))
    all_routes = []
    
    iteration = 0
    while unrouted:
        iteration += 1
        print(f"\n开始构建第 {iteration} 条路径...")
        
        # 步骤1: 初始化新路径
        if initialize_with == "farthest":
            # 选择距离仓库最远的未分配客户
            init_customer = max(unrouted, key=lambda x: D[0, x])
        elif initialize_with == "closest":
            # 选择距离仓库最近的未分配客户
            init_customer = min(unrouted, key=lambda x: D[0, x])
        else:
            # 随机选择一个客户
            init_customer = random.choice(list(unrouted))
        
        # 创建初始路径: 仓库 -> 客户 -> 仓库
        current_route = [0, init_customer, 0]
        current_demand = demands[init_customer]
        current_distance = D[0, init_customer] + D[init_customer, 0]
        
        unrouted.remove(init_customer)
        
        print(f"  初始客户: {init_customer}, 初始距离: {current_distance:.2f}")
        
        # 步骤2: 构建当前路径
        improved = True
        while improved and unrouted:
            improved = False
            best_candidate = None
            best_position = -1
            best_saving = float('inf')
            
            # 对于每个未分配的客户
            for customer in list(unrouted):
                # 检查容量约束
                if current_demand + demands[customer] > capacity:
                    continue
                
                # 找到最佳的插入位置
                for pos in range(1, len(current_route)):
                    i = current_route[pos-1]  # 前一个节点
                    j = current_route[pos]    # 后一个节点
                    
                    # 计算插入成本
                    insertion_cost = D[i, customer] + D[customer, j] - D[i, j]
                    
                    # 计算节约（相比于单独服务）
                    saving = insertion_cost - (D[0, customer] + D[customer, 0])
                    
                    # 使用参数化的插入准则
                    strain, _ = parametrized_insertion_criteria(D, i, customer, j, lambda_mul, mu_mul)
                    
                    if (saving < best_saving) or (abs(saving - best_saving) < 1e-9 and strain > best_strain):
                        best_saving = saving
                        best_strain = strain
                        best_candidate = customer
                        best_position = pos
                        best_insertion_cost = insertion_cost
            
            if best_candidate is not None:
                # 检查插入是否会恶化解
                if best_saving < 0:  # 如果插入能节省距离
                    # 执行插入
                    current_route.insert(best_position, best_candidate)
                    current_demand += demands[best_candidate]
                    current_distance += best_insertion_cost
                    unrouted.remove(best_candidate)
                    
                    print(f"  插入客户 {best_candidate} 在位置 {best_position}, 新距离: {current_distance:.2f}")
                    improved = True
                else:
                    break
            else:
                # 没有可行的插入，开始下一条路径
                break
        
        all_routes.append(current_route)
        print(f"  路径完成: {current_route}, 总需求: {current_demand}, 距离: {current_distance:.2f}")
    
    total_distance = calculate_total_distance(all_routes, D)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n" + "=" * 70)
    print("算法完成！")
    print(f"总路径数: {len(all_routes)}")
    print(f"总距离: {total_distance:.2f}")
    print(f"运行时间: {elapsed_time:.3f} 秒")
    
    print(f"\n详细路径:")
    for i, route in enumerate(all_routes):
        route_demand = sum(demands[node] for node in route[1:-1])
        route_length = calculate_route_length(route, D)
        
        print(f"  路径{i+1}: {route} (需求: {route_demand}/{capacity}, "
              f"距离: {route_length:.2f}")
    
    return all_routes, total_distance

def parallel_insertion_heuristic(points, demands, capacity, D, 
                                lambda_mul=2.0, mu_mul=1.0,
                                num_parallel_routes="auto"):
    """
    并行插入启发式算法
    
    参数:
        num_parallel_routes: 并行构建的路径数，"auto" 表示自动确定
    """
    n = len(points)
    start_time = time.time()
    
    # 未分配客户集合
    unrouted = set(range(1, n))
    
    # 确定并行路径数
    if num_parallel_routes == "auto":
        # 自动估计：最少需要的车辆数
        min_vehicles = int(np.ceil(sum(demands) / capacity))
        num_parallel_routes = min(min_vehicles, len(unrouted))
    else:
        num_parallel_routes = min(num_parallel_routes, len(unrouted))
    
    print(f"并行构建 {num_parallel_routes} 条路径")
    
    # 初始化多条路径
    routes = []
    route_demands = []
    route_distances = []
    
    # 为每条路径选择初始客户（选择距离仓库最远的客户）
    sorted_customers = sorted(unrouted, key=lambda x: D[0, x], reverse=True)
    
    for i in range(num_parallel_routes):
        if not sorted_customers:
            break
            
        customer = sorted_customers[0]
        sorted_customers = [c for c in sorted_customers if c != customer]
        
        route = [0, customer, 0]
        routes.append(route)
        route_demands.append(demands[customer])
        route_distances.append(D[0, customer] + D[customer, 0])
        unrouted.remove(customer)
    
    print(f"初始化完成，创建了 {len(routes)} 条路径")
    
    # 并行插入过程
    iteration = 0
    while unrouted:
        iteration += 1
        if iteration % 10 == 0:
            print(f"  迭代 {iteration}, 剩余未分配客户: {len(unrouted)}")
        
        # 为每个未分配客户找到最佳插入位置：遍历所有位置，插入成本最小原则
        candidate_insertions = []  # (saving, strain, customer, route_idx, position)
        
        for customer in list(unrouted):
            for route_idx, route in enumerate(routes):
                # 检查容量约束
                if route_demands[route_idx] + demands[customer] > capacity:
                    continue
                
                # 找到最佳插入位置
                best_saving = float('inf')
                best_strain = -float('inf')
                best_position = -1
                
                for pos in range(1, len(route)):
                    i = route[pos-1]
                    j = route[pos]
                    
                    # 计算插入成本
                    insertion_cost = D[i, customer] + D[customer, j] - D[i, j]
                    
                    # 计算节约
                    saving = insertion_cost - (D[0, customer] + D[customer, 0])
                    
                    # 使用参数化的插入准则
                    strain, _ = parametrized_insertion_criteria(D, i, customer, j, lambda_mul, mu_mul)
                    
                    if (saving < best_saving) or (abs(saving - best_saving) < 1e-9 and strain > best_strain):
                        best_saving = saving
                        best_strain = strain
                        best_position = pos
                        best_insertion_cost = insertion_cost
                
                if best_position != -1:
                    candidate_insertions.append((best_saving, -best_strain, customer, route_idx, best_position, best_insertion_cost))
        
        if not candidate_insertions:
            # 没有可行的插入，创建新路径
            if unrouted:
                customer = max(unrouted, key=lambda x: D[0, x])
                route = [0, customer, 0]
                routes.append(route)
                route_demands.append(demands[customer])
                route_distances.append(D[0, customer] + D[customer, 0])
                unrouted.remove(customer)
                print(f"  创建新路径，客户 {customer}")
            continue
        
        # 选择最佳插入（节约值最小，strain最大）
        candidate_insertions.sort(key=lambda x: (x[0], x[1]))
        
        for saving, _, customer, route_idx, position, insertion_cost in candidate_insertions:
            if customer not in unrouted:
                continue
                
            if saving < 0:  # 插入能节省距离
                # 执行插入
                routes[route_idx].insert(position, customer)
                route_demands[route_idx] += demands[customer]
                route_distances[route_idx] += insertion_cost
                unrouted.remove(customer)
                break
    
    # 计算总距离
    total_distance = calculate_total_distance(routes, D)
    elapsed_time = time.time() - start_time
    
    # 输出结果
    print(f"\n" + "=" * 70)
    print("算法完成！")
    print(f"总路径数: {len(routes)}")
    print(f"总距离: {total_distance:.2f}")
    print(f"运行时间: {elapsed_time:.3f} 秒")
    
    print(f"\n详细路径:")
    for i, route in enumerate(routes):
        route_demand = sum(demands[node] for node in route[1:-1])
        route_length = calculate_route_length(route, D)
        print(f"  路径{i+1}: {route} (需求: {route_demand}/{capacity}, "
              f"距离: {route_length:.2f}")
    
    return routes, total_distance

def generate_random_problem(num_customers=20, area_size=100, max_demand=20, vehicle_capacity=80):
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

def main():
    num_customers = 20
    area_size = 100
    max_demand = 20
    vehicle_capacity = 80
    
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
    
    # 运行顺序插入算法
    seq_routes, seq_distance = sequential_insertion_heuristic(
        points, demands, capacity, D,
        lambda_mul=2.0,  # Solomon (1987) 推荐的值
        mu_mul=1.0,
        initialize_with="farthest"
    )
    
    # 运行并行插入算法
    print(f"\n" + "="*60)
    print("运行并行插入启发式算法...")
    
    par_routes, par_distance = parallel_insertion_heuristic(
        points, demands, capacity, D,
        lambda_mul=2.0,
        mu_mul=1.0,
        num_parallel_routes="auto"
    )
    
    # 结果对比
    print(f"\n" + "="*60)
    print("结果对比:")
    print("=" * 60)
    
    print(f"顺序插入算法:")
    print(f"  总距离: {seq_distance:.2f}")
    print(f"  车辆数: {len(seq_routes)}")
    
    print(f"\n并行插入算法:")
    print(f"  总距离: {par_distance:.2f}")
    print(f"  车辆数: {len(par_routes)}")
    
    if par_distance < seq_distance:
        improvement = (seq_distance - par_distance) / seq_distance * 100
        print(f"\n并行插入算法优于顺序插入算法: 改善 {improvement:.2f}%")
    elif seq_distance < par_distance:
        improvement = (par_distance - seq_distance) / par_distance * 100
        print(f"\n顺序插入算法优于并行插入算法: 改善 {improvement:.2f}%")
    else:
        print(f"\n两种算法找到相同质量的解")
    
    # # 算法参数调优测试
    
    # lambda_values = [0.0, 0.5, 1.0, 1.5, 2.0]
    # mu_values = [0.0, 0.5, 1.0, 1.5]
    
    # best_distance = float('inf')
    # best_params = None
    # best_routes = None
    
    # for lm in lambda_values:
    #     for mm in mu_values:
    #         print(f"\n测试 lambda={lm}, mu={mm}...")
    #         test_routes, test_distance = sequential_insertion_heuristic(
    #             points, demands, capacity, D,
    #             lambda_mul=lm,
    #             mu_mul=mm,
    #             initialize_with="farthest"
    #         )
            
    #         if test_distance < best_distance:
    #             best_distance = test_distance
    #             best_params = (lm, mm)
    #             best_routes = test_routes
    
    # print(f"\n最佳参数组合: lambda={best_params[0]}, mu={best_params[1]}")
    # print(f"最佳距离: {best_distance:.2f}")
    
    
    
    return {
        'points': points,
        'demands': demands,
        'capacity': capacity,
        'seq_routes': seq_routes,
        'seq_distance': seq_distance,
        'par_routes': par_routes,
        'par_distance': par_distance,
        # 'best_params': best_params,
        # 'best_distance': best_distance,
        # 'best_routes': best_routes
    }

if __name__ == "__main__":
    result = main()