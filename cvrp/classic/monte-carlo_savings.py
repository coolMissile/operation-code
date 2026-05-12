#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Monte Carlo Savings算法实现
    Monte Carlo Savings algorithm (OLIVEIRA, 2014)
    https://github.com/RomuloOliveira/monte-carlo-cvrp.git改编
    
"""

import numpy as np
import random
import time
import matplotlib.pyplot as plt

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

def clarke_wright_with_savings_list(points, demands, capacity, D, savings_list):
    """
    使用给定的节约值列表执行CW算法
    
    Args:
        savings_list: 排序后的节约值列表 [(i,j), ...]
    """
    n = len(points)
    
    routes = [[0, i, 0] for i in range(1, n)]
    route_demands = demands[1:] 
    customer_to_route = {i: i-1 for i in range(1, n)}

    for customer_i, customer_j in savings_list:
        route_i_idx = customer_to_route.get(customer_i)
        route_j_idx = customer_to_route.get(customer_j)
        
        if route_i_idx is None or route_j_idx is None:
            continue
            
        if route_i_idx == route_j_idx:
            continue
            
        route_i = routes[route_i_idx]
        route_j = routes[route_j_idx]
        
        if route_i is None or route_j is None:
            continue
       
        total_demand = route_demands[route_i_idx] + route_demands[route_j_idx]
        if total_demand > capacity:
            continue
        
        i_is_endpoint = (route_i[1] == customer_i or route_i[-2] == customer_i)
        j_is_endpoint = (route_j[1] == customer_j or route_j[-2] == customer_j)
        
        if not (i_is_endpoint and j_is_endpoint):
            continue
       # 和c-w savings相同，四种情况
        new_route = None
        
        # 情况1: i尾 + j头
        if route_i[-2] == customer_i and route_j[1] == customer_j:
            new_route = route_i[:-1] + route_j[1:]
        
        # 情况2: i头 + j尾
        elif route_i[1] == customer_i and route_j[-2] == customer_j:
            new_route = route_j[:-1] + route_i[1:]
        
        # 情况3: i尾 + j尾
        elif route_i[-2] == customer_i and route_j[-2] == customer_j:
            new_route = route_i[:-1] + route_j[-2:0:-1] + [0]
        
        # 情况4: i头 + j头
        elif route_i[1] == customer_i and route_j[1] == customer_j:
            new_route = [0] + route_i[:0:-1] + route_j[1:]
        
        if not new_route:
            continue
        
        routes[route_i_idx] = new_route
        routes[route_j_idx] = None
        route_demands[route_i_idx] = total_demand
        route_demands[route_j_idx] = 0
        
        for customer in new_route[1:-1]: 
            customer_to_route[customer] = route_i_idx
        
        for customer in route_j[1:-1]:
            if customer in customer_to_route and customer_to_route[customer] == route_j_idx:
                customer_to_route[customer] = route_i_idx
    
    # 清理空路径
    valid_routes = [route for route in routes if route is not None]
    return valid_routes

def monte_carlo_savings_algorithm(points, demands, capacity, D, 
                                  lambda_p=0.05, simulations=100, timeout=60):
    """
    Monte Carlo Savings算法主函数
    
    Args:
        lambda_p: 随机扰动幅度，默认0.05（±5%）
        simulations: 蒙特卡洛模拟次数，默认100
        timeout: 超时时间（秒），默认60
    """
    n = len(points)
    
    print("=" * 70)
    print("Monte Carlo Savings 算法 (OLIVEIRA, 2014)")
    print("=" * 70)
    print(f"参数: lambda_p={lambda_p}, 模拟次数={simulations}, 超时={timeout}秒")
    
    start_time = time.time()
    
    # 存储所有模拟结果
    all_solutions = []
    all_distances = []
    
    # 记录最佳解
    best_routes = None
    best_distance = float('inf')
    best_iteration = -1
    
    # 基础节约值计算（用于扰动）
    base_savings = {}
    for i in range(1, n):
        for j in range(i+1, n):
            saving = D[i, 0] + D[0, j] - D[i, j]
            base_savings[(i, j)] = saving
    
    print(f"\n基础节约值计算完成，共有 {len(base_savings)} 个客户对")
    print("开始蒙特卡洛模拟...")
    
    for sim in range(simulations):
        if time.time() - start_time > timeout:
            print(f"超时，已运行 {sim} 次模拟")
            break
        
        # 步骤1: 生成带扰动的节约值列表
        perturbed_savings = {}
        
        for (i, j), base_saving in base_savings.items():
            # 添加随机扰动
            p = random.uniform(-lambda_p, lambda_p)
            perturbed_saving = base_saving + (base_saving * p)
            perturbed_savings[(i, j)] = perturbed_saving
        
        # 步骤2: 按扰动后节约值降序排序
        sorted_savings = sorted(perturbed_savings.items(), 
                                key=lambda x: x[1], reverse=True)
        savings_list = [pair for pair, _ in sorted_savings]
        
        # 步骤3: 使用这个随机排序执行CW算法
        routes = clarke_wright_with_savings_list(points, demands, capacity, D, savings_list)
        distance = calculate_total_distance(routes, D)
        
        # 记录结果
        all_solutions.append(routes)
        all_distances.append(distance)
        
        # 更新最佳解
        if distance < best_distance:
            best_distance = distance
            best_routes = routes
            best_iteration = sim
        
        if (sim + 1) % 20 == 0:
            print(f"  已完成 {sim+1}/{simulations} 次模拟，当前最佳: {best_distance:.2f}")
    
    # 计算统计信息
    avg_distance = np.mean(all_distances) if all_distances else 0
    std_distance = np.std(all_distances) if len(all_distances) > 1 else 0
    min_distance = min(all_distances) if all_distances else 0
    max_distance = max(all_distances) if all_distances else 0
    
    print(f"\n模拟完成，实际运行 {len(all_distances)} 次")
    print(f"平均路径长度: {avg_distance:.2f} ± {std_distance:.2f}")
    print(f"最小路径长度: {min_distance:.2f}")
    print(f"最大路径长度: {max_distance:.2f}")
    print(f"最佳解在第 {best_iteration+1} 次模拟中找到")
    
    # 显示最佳解详情
    print(f"\n最佳解详情 (长度: {best_distance:.2f}):")
    for i, route in enumerate(best_routes):
        route_demand = sum(demands[node] for node in route[1:-1])
        route_length = calculate_route_length(route, D)
        print(f"  车辆{i+1}: {route} (需求: {route_demand}/{capacity}, 距离: {route_length:.2f})")
    
    elapsed_time = time.time() - start_time
    print(f"\n总运行时间: {elapsed_time:.2f} 秒")
    print(f"平均每次模拟时间: {elapsed_time/len(all_distances):.4f} 秒" if all_distances else "")
    
    return best_routes, best_distance, {
        'avg_distance': avg_distance,
        'std_distance': std_distance,
        'min_distance': min_distance,
        'max_distance': max_distance,
        'all_distances': all_distances,
        'best_iteration': best_iteration
    }

def generate_random_problem(num_customers=10, area_size=100, max_demand=20, vehicle_capacity=50):
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
    # 参数设置
    num_customers = 20
    area_size = 100
    max_demand = 20
    vehicle_capacity = 80
    
    # Monte Carlo 参数
    lambda_p = 0.05
    simulations = 100
    timeout = 30  # 秒
    
    print(f"\n生成随机CVRP问题...")
    print(f"  客户数量: {num_customers}")
    print(f"  区域大小: {area_size}x{area_size}")
    print(f"  最大需求: {max_demand}")
    print(f"  车辆容量: {vehicle_capacity}")
    print(f"  Monte Carlo参数: lambda_p={lambda_p}, 模拟次数={simulations}")
    
    # 生成问题实例
    points, demands, capacity = generate_random_problem(
        num_customers, area_size, max_demand, vehicle_capacity
    )
    
    print(f"理论最小车辆数: {sum(demands)/capacity:.1f}")
    
    # 计算距离矩阵
    D = calculate_distance_matrix(points)
    
    # # 运行标准CW算法作为对比基准
    # print(f"\n" + "="*60)
    # print("运行标准Clarke-Wright算法作为基准...")
    
    # # 先运行一次标准CW（相当于lambda_p=0的Monte Carlo）
    # base_savings = []
    # n = len(points)
    # for i in range(1, n):
    #     for j in range(i+1, n):
    #         saving = D[i, 0] + D[0, j] - D[i, j]
    #         base_savings.append((saving, i, j))
    # base_savings.sort(reverse=True, key=lambda x: x[0])
    # base_savings_list = [(i, j) for saving, i, j in base_savings]
    
    # base_routes = clarke_wright_with_savings_list(points, demands, capacity, D, base_savings_list)
    # base_distance = calculate_total_distance(base_routes, D)
    
    # print(f"标准CW算法结果:")
    # print(f"  车辆数: {len(base_routes)}")
    # print(f"  总距离: {base_distance:.2f}")
    
    # # 运行Monte Carlo Savings算法
    best_routes, best_distance, stats = monte_carlo_savings_algorithm(
        points, demands, capacity, D, 
        lambda_p=lambda_p, 
        simulations=simulations, 
        timeout=timeout
    )
    
    # 对比结果
    # print(f"\n" + "="*60)
    # print("结果对比:")
    # print("=" * 60)
    # print(f"标准Clarke-Wright算法:")
    # print(f"  总距离: {base_distance:.2f}")
    # print(f"  车辆数: {len(base_routes)}")
    
    print(f"\nMonte Carlo Savings算法:")
    print(f"  最佳总距离: {best_distance:.2f}")
    print(f"  车辆数: {len(best_routes)}")
    # print(f"  改进比例: {(base_distance - best_distance)/base_distance*100:.2f}%")
    
    # if best_distance < base_distance:
    #     print(f"Monte Carlo Savings 算法找到了更好的解！")
    # elif abs(best_distance - base_distance) < 0.01:
    #     print(f"两种算法找到相同质量的解")
    # else:
    #     print(f"标准CW算法找到了更好的解")
    
    print(f"\n" + "="*60)
    print("求解完成！")
    print("=" * 60)
    
    return {
        'points': points,
        'demands': demands,
        'capacity': capacity,
        # 'base_routes': base_routes,
        # 'base_distance': base_distance,
        'best_routes': best_routes,
        'best_distance': best_distance,
        'stats': stats
    }

if __name__ == "__main__":
    result = main()
    