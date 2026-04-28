#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""基础节约算法实现
    从verypy的cawlip_savings.py删掉2-opt优化，简化
"""

import numpy as np
import matplotlib.pyplot as plt
import random

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

def calculate_savings(D):
    """计算Clark-Wright节约值"""
    n = len(D)
    savings = []
    
    # 对于每对客户(i,j)，计算节约值
    for i in range(1, n):  # 从1开始，跳过仓库0
        for j in range(i+1, n):
            # 节约值 = d(i,0) + d(0,j) - d(i,j)
            saving = D[i, 0] + D[0, j] - D[i, j]
            savings.append((saving, i, j))
    
    # 按节约值降序排序
    savings.sort(reverse=True, key=lambda x: x[0])
    return savings

def basic_savings_algorithm(points, demands, capacity, D):
    """
    基础Clark-Wright节约算法实现
    
    参数:
    points: 点坐标列表，points[0]是仓库
    demands: 需求列表，demands[0]是仓库需求(0)
    capacity: 车辆容量
    D: 距离矩阵
    """
    n = len(points)
    
    print("=" * 60)
    print("Clark-Wright 节约算法 (1964)")
    print("=" * 60)
    
    # 步骤1: 初始化为每个客户单独一条路径
    routes = [[0, i, 0] for i in range(1, n)]
    route_demands = demands[1:]  # 每条路径的需求
    
    print(f"\n初始化: 每个客户单独一条路径")
    print(f"初始路径数: {len(routes)}")
    for i, route in enumerate(routes):
        demand = demands[route[1]]
        cost = D[0, route[1]] + D[route[1], 0]
        print(f"  路径{i+1}: {route} (需求: {demand}, 距离: {cost:.2f})")
    
    # 步骤2: 计算节约值
    savings_list = calculate_savings(D)
    print(f"\n计算了 {len(savings_list)} 个节约值")
    print(f"前5个节约值:")
    for i in range(min(5, len(savings_list))):
        s, c1, c2 = savings_list[i]
        print(f"  s({c1},{c2}) = {s:.2f}")
    
    # 步骤3: 按节约值从大到小尝试合并路径
    print(f"\n开始合并路径...")
    
    # 用于跟踪每个客户所在的路径索引
    customer_to_route = {i: i-1 for i in range(1, n)}
    
    for saving, customer_i, customer_j in savings_list:
        # 获取客户i和j所在的路径索引
        route_i_idx = customer_to_route.get(customer_i)
        route_j_idx = customer_to_route.get(customer_j)
        
        if route_i_idx is None or route_j_idx is None:
            continue
            
        if route_i_idx == route_j_idx:
            continue
            
        route_i = routes[route_i_idx]
        route_j = routes[route_j_idx]
        
        # 检查合并后的需求是否超过容量
        total_demand = route_demands[route_i_idx] + route_demands[route_j_idx]
        if total_demand > capacity:
            continue
        
        # 检查客户i和j是否在各自路径的端点
        i_is_endpoint = (route_i[1] == customer_i or route_i[-2] == customer_i)
        j_is_endpoint = (route_j[1] == customer_j or route_j[-2] == customer_j)
        
        if not (i_is_endpoint and j_is_endpoint):
            continue
        
        # 确定如何连接两条路径
        new_route = []
        
        # 情况1: i在route_i的末尾，j在route_j的开头
        if route_i[-2] == customer_i and route_j[1] == customer_j:
            new_route = route_i[:-1] + route_j[1:]  # 0...i + j...0
        
        # 情况2: i在route_i的开头，j在route_j的末尾
        elif route_i[1] == customer_i and route_j[-2] == customer_j:
            new_route = route_j[:-1] + route_i[1:]  # 0...j + i...0
        
        # 情况3: i在route_i的末尾，j在route_j的末尾
        elif route_i[-2] == customer_i and route_j[-2] == customer_j:
            new_route = route_i[:-1] + route_j[-2:0:-1] + [0]  # 0...i + 反转的j...0
        
        # 情况4: i在route_i的开头，j在route_j的开头
        elif route_i[1] == customer_i and route_j[1] == customer_j:
            new_route = [0] + route_i[:0:-1] + route_j[1:]  # 0 + 反转的i...0 + j...0
        
        if not new_route:
            continue
        
        # 计算合并后的路径成本
        new_cost = 0
        for k in range(len(new_route)-1):
            new_cost += D[new_route[k], new_route[k+1]]
        
        # 计算旧的路径成本
        old_cost_i = 0
        for k in range(len(route_i)-1):
            old_cost_i += D[route_i[k], route_i[k+1]]
        
        old_cost_j = 0
        for k in range(len(route_j)-1):
            old_cost_j += D[route_j[k], route_j[k+1]]
        
        cost_saving = (old_cost_i + old_cost_j) - new_cost
        
        print(f"  合并客户 {customer_i} 和 {customer_j}: 节约 {cost_saving:.2f}")
        print(f"    路径变化: {route_i} + {route_j} -> {new_route}")
        print(f"    需求: {route_demands[route_i_idx]} + {route_demands[route_j_idx]} = {total_demand}")
        
        # 更新数据
        routes[route_i_idx] = new_route
        routes[route_j_idx] = None  # 标记为已删除
        route_demands[route_i_idx] = total_demand
        route_demands[route_j_idx] = 0
        
        # 更新客户到路径的映射
        for customer in new_route[1:-1]:  # 不包括首尾的0
            customer_to_route[customer] = route_i_idx
        
        # 标记被删除路径的客户
        for customer in route_j[1:-1]:
            if customer in customer_to_route and customer_to_route[customer] == route_j_idx:
                customer_to_route[customer] = route_i_idx
    
    # 移除被标记为删除的路径
    valid_routes = [route for route in routes if route is not None]
    valid_demands = [demand for i, demand in enumerate(route_demands) if routes[i] is not None]
    
    # 计算结果
    total_distance = 0
    print(f"\n最终路径:")
    for i, route in enumerate(valid_routes):
        route_cost = 0
        for k in range(len(route)-1):
            route_cost += D[route[k], route[k+1]]
        total_distance += route_cost
        
        route_demand = 0
        for node in route[1:-1]:  # 不包括仓库
            route_demand += demands[node]
        
        print(f"  路径{i+1}: {route} (需求: {route_demand}/{capacity}, 距离: {route_cost:.2f})")
    
    print(f"\n总结:")
    print(f"  总车辆数: {len(valid_routes)}")
    print(f"  总行驶距离: {total_distance:.2f}")
    print(f"  总客户需求: {sum(demands)}")
    print(f"  平均车辆利用率: {sum(demands)/(len(valid_routes)*capacity)*100:.1f}%")
    
    return valid_routes, total_distance

def generate_random_problem(num_customers=10, area_size=100, max_demand=20, vehicle_capacity=50):
    """生成随机VRP问题"""
    # 生成仓库位置
    depot = (area_size/2, area_size/2)
    
    # 生成客户位置
    customers = []
    for i in range(num_customers):
        x = random.uniform(0, area_size)
        y = random.uniform(0, area_size)
        customers.append((x, y))
    
    # 生成需求
    demands = [0]  
    for i in range(num_customers):
        demands.append(random.randint(1, max_demand))
    
    # 所有点（仓库+客户）
    all_points = [depot] + customers
    
    return all_points, demands, vehicle_capacity



def main():
   
    # 生成随机问题
    num_customers = 10
    area_size = 100
    max_demand = 20
    vehicle_capacity = 50
    
    print(f"\n生成随机CVRP问题...")
    print(f"  客户数量: {num_customers}")
    print(f"  区域大小: {area_size}x{area_size}")
    print(f"  最大需求: {max_demand}")
    print(f"  车辆容量: {vehicle_capacity}")
    
    points, demands, capacity = generate_random_problem(
        num_customers, area_size, max_demand, vehicle_capacity
    )
    
    # 显示客户信息
    print(f"\n客户信息:")
    print("  客户 |  X坐标  |  Y坐标  | 需求")
    print("  " + "-" * 30)
    for i in range(1, len(points)):
        print(f"  {i:3d} | {points[i][0]:6.1f} | {points[i][1]:6.1f} | {demands[i]:3d}")
    
    print(f"\n仓库位置: ({points[0][0]:.1f}, {points[0][1]:.1f})")
    print(f"总需求: {sum(demands)}")
    print(f"最小所需车辆数: {sum(demands)/capacity:.1f}")
    
    # 计算距离矩阵
    D = calculate_distance_matrix(points)
    
    # 运行节约算法
    print(f"\n" + "="*60)
    routes, total_distance = basic_savings_algorithm(points, demands, capacity, D)
    
    
    
    print(f"\n" + "="*60)
    print("求解完成！")
    print("=" * 60)
    
    return routes, total_distance

if __name__ == "__main__":
    routes, total_distance = main()