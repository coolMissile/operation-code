#由verypy的sweep.py改写
#!/usr/bin/env python
# -*- coding: utf-8 -*-
###############################################################################
""" This file is a part of the VeRyPy classical vehicle routing problem
heuristic library and provides an implementation of the general Sweep approach
of Gillett and Miller (1974) and Wren & Holliday (1972). This basic 
implementation does not include improvement heuristics, but is written in a
way that it can be extended with online and post-optimization procedures.

The script is callable and can be used as a standalone solver for TSPLIB
formatted CVRPs. It has extensive dependencies: a TSP solver (the built in
local search solver can be used), and numpy and scipy for reading and preparing
the problem instance.

使用极坐标，顺时针/逆时针扫描，寻找最近能运载的
"""
###############################################################################

# Written in Python 2.7, but try to maintain Python 3+ compatibility
from __future__ import print_function
from __future__ import division
    
from math import pi
import logging

import numpy as np

#  the ordered property is used by wren_holliday (the routes are built in the
#  order the nodes are added during the sweep).
from collections import OrderedDict
import itertools

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# 简化版本的工具函数
def objf(route, D):
    """计算路径的总距离"""
    total = 0.0
    for i in range(len(route)-1):
        total += D[route[i], route[i+1]]
    return total

def without_empty_routes(sol):
    """移除空路径"""
    result = []
    i = 0
    while i < len(sol):
        if i+1 < len(sol) and sol[i] == 0 and sol[i+1] == 0:
            i += 1
        else:
            result.append(sol[i])
        i += 1
    return result

def is_better_sol(best_f, best_K, sol_f, sol_K, minimize_K=False):
    """判断新解是否更好"""
    if best_f is None:
        return True
    if minimize_K:
        if sol_K < best_K:
            return True
        elif sol_K == best_K and sol_f < best_f:
            return True
    else:
        if sol_f < best_f:
            return True
    return False

# 简化版本的RouteData
class RouteData:
    def __init__(self, route, cost=0, demand=0, node_set=None):
        self.route = route
        self.cost = cost
        self.demand = demand
        self.node_set = OrderedDictSet() if node_set is None else node_set
        
    def __repr__(self):
        return f"RouteData(route={self.route}, cost={self.cost:.2f}, demand={self.demand})"

# 简化版本的OrderedSet
class OrderedDictSet:
    def __init__(self, iterable=None):
        self._dict = OrderedDict()
        if iterable:
            for item in iterable:
                self.add(item)
    
    def add(self, item):
        self._dict[item] = None
    
    def remove(self, item):
        del self._dict[item]
    
    def __contains__(self, item):
        return item in self._dict
    
    def __iter__(self):
        return iter(self._dict.keys())
    
    def __len__(self):
        return len(self._dict)
    
    def __repr__(self):
        return f"OrderedDictSet({list(self._dict.keys())})"
    
    def copy(self):
        new_set = OrderedDictSet()
        new_set._dict = OrderedDict(self._dict)
        return new_sets

# 配置参数
C_EPS = 1e-6
S_EPS = 1e-6

CLOSEST_TO_DEPOT = 0
SMALLEST_ANGLE = -1
BEST_ALTERNATIVE = -2

PHI = 0
RHO = 1
NODE = 2

def cart2pol(x, y):
    """ helper for that converts x,y to polar coords.
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)

def bisect_angle(angle1, angle2, ratio=0.5, direction=1):
    """ A helper function that gives an angle between angle1 and angle2
    (in rads) based on the bisect ratio. Default gives the angle that bisects
    the sector in the middle.
    """
    while direction*angle2<direction*angle1:
        angle2+=direction*2*pi            
    return angle1-(angle1-angle2)*ratio

def _step(current, inc, max_val):
    current+=inc
    if current>max_val:
        current = 0
    if current<0:
        # reverse direction
        current = max_val
    return current

def _ensure_L_feasible(D, d, C, L, current_route,
                       L_feasible_sweep_pos, step_inc, max_sweep_idx,
                       routed, sweep_pos_to_node_idx, routing_algo):
    """ This makes sure the current route in satisfies the
        route length/cost/duration constraint (L). We follow Gillett & Miller 
        (1974) where nodes are removed from the route set until the route
        cost satisfies the L constraint.
        
        Potentially modifies current_route and routed"""
    pos_to_remove = L_feasible_sweep_pos
    while True:
        if current_route.cost-S_EPS<=L:
            break
        else:
            pos_to_remove = _step(pos_to_remove, -step_inc, max_sweep_idx)   
            node_to_remove = sweep_pos_to_node_idx(pos_to_remove)
            if node_to_remove in current_route.node_set:
                current_route.node_set.remove(node_to_remove)       
                routed[node_to_remove] = False
                tsp_sol, tsp_cost = routing_algo(D, list(current_route.node_set))
                # update current route
                current_route.route = tsp_sol
                current_route.cost = tsp_cost
                if C:
                    current_route.demand-=d[node_to_remove]
                
                # also the sweep goes back one step
                L_feasible_sweep_pos = pos_to_remove   
                
                log.debug(f"L constraint was violated, removed n{node_to_remove} from the route set")
                log.debug(f"Got TSP solution {tsp_sol} ({tsp_cost:.2f})")
            
    return L_feasible_sweep_pos

def do_one_sweep(N, D, d, C, L, routing_algo,
                  sweep, start, step_inc,
                  generate_alternative_first_routes = False,
                  intra_route_callback=None, inter_route_callback=None,
                  callback_data=None):
    """ This function does one full circle starting from start index of the 
    sweep and proceeds to add the nodes one by one in the direction indicated 
    by the step_inc parameter. A new route is started if either of the
    constraints constraints C and L (L can be None) are violated.

    If generate_alternative_first_routes is set, every possible route until 
    C OR L constraints are generated and then the sweep terminates.
    This is used by some constuction heuristics to build potential routes.
    
    The generated routes are TSP optimized using routing_algo and may be 
    further optimized with intra and inter route  improvement callbacks.""" 
    
    # make sure all before the start node have larger angles than the start node
    # and that they are of range [0-2*pi]
    if intra_route_callback or inter_route_callback:
        sweep_phis = np.copy( sweep[0] )
        sweep_phis-=(pi+sweep_phis[start]) # the first is at -pi
        sweep_phis[start-step_inc::-step_inc]+=2*pi
        sweep_phis = -1*sweep_phis
    
    sweep_pos_to_node_idx = lambda idx: int(sweep[2,idx])
    max_sweep_idx = len(sweep[2])-1
    
    int_sweep = list(sweep[2].astype(int))
    log.info(f"Sweep node order {int_sweep[start:]+int_sweep[:start]}")

    # Routes
    routes = [] 
    routed = [False]*N
    routed[0] = True
    routed_cnt = 0
    total_to_route = len(sweep[0])
    
    # Emerging route
    current_route = RouteData([0])
    current_route.node_set = OrderedDictSet([0])
    current_route_cost_upper_bound = 0
    route_complete = False
    
    # Backlog
    blocked_nodes = set()
    
    # THE MAIN SWEEP LOOP
    sweep_pos = start
    sweep_node = sweep_pos_to_node_idx(sweep_pos)  
    while True:
        if sweep_node:
            prev_pos = _step(sweep_pos, -step_inc, max_sweep_idx)
            next_pos = _step(sweep_pos, step_inc, max_sweep_idx)
            prev_ray = bisect_angle(sweep[0][prev_pos], sweep[0][sweep_pos], direction=step_inc) 
            next_ray = bisect_angle(sweep[0][sweep_pos], sweep[0][next_pos], direction=step_inc) 
            log.debug(f"Considering n{sweep_node} between rays {prev_ray:.2f}, {next_ray:.2f}")
        
        # 检查是否应该结束当前路径
        if not route_complete and C:
            would_break_C_ctr = current_route.demand+d[sweep_node]-C_EPS>C
            route_complete = would_break_C_ctr            
        if not route_complete and L and current_route_cost_upper_bound>L:
            tsp_sol, tsp_cost = routing_algo(D, list(current_route.node_set)+[sweep_node]) 
            current_route_cost_upper_bound = tsp_cost
            would_break_L_ctr = tsp_cost-S_EPS>L
            route_complete = would_break_L_ctr
            
        route_interesting = generate_alternative_first_routes and len(current_route.node_set)>1
        
        if route_complete or route_interesting:  
            tsp_sol, tsp_cost = routing_algo(D, list(current_route.node_set))
            current_route.route = tsp_sol
            current_route.cost = tsp_cost
            current_route_cost_upper_bound = current_route.cost
            
            if route_complete:
                log.debug(f"Route {list(current_route.node_set)} full.")
            log.debug(f"Got TSP solution {tsp_sol} ({tsp_cost:.2f})")
                     
            if L:
                L_feasible_pos = _ensure_L_feasible(D, d, C, L, 
                    current_route, sweep_pos, step_inc, max_sweep_idx, routed, 
                    sweep_pos_to_node_idx, routing_algo)
                current_route_cost_upper_bound = current_route.cost
                
                if L_feasible_pos!=sweep_pos:
                    sweep_pos = L_feasible_pos  
                    if sweep_node!=None:
                        sweep_node = sweep_pos_to_node_idx(sweep_pos)
                    route_complete = True
                    route_interesting = False
                
            if sweep_node!=None and intra_route_callback:                
                current_route, included_nodes, ignored_nodes, route_complete = \
                    intra_route_callback( current_route, callback_data,
                                          sweep[1], sweep_phis,
                                          sweep_pos, step_inc, routed )
                current_route_cost_upper_bound = current_route.cost
                
                log.debug(f"Improved the route to {current_route.route} ({current_route.cost:.2f})")
                
                for rn in included_nodes:
                    routed[rn]=True  
                for lon in ignored_nodes:
                    routed[lon] = False
                    log.debug(f"Block n{lon} for now")
                    blocked_nodes.add(lon)
                    if lon==sweep_node:
                        sweep_node = None
                
            if route_complete:
                log.info(f"Route completed {current_route.route}")
                
                route_customer_cnt = len(current_route.node_set)
                if route_customer_cnt>1:
                    routed_cnt+=route_customer_cnt-1
                    routes.append( current_route )
                    
                if routed_cnt>=total_to_route:
                    log.info(f"Last route completed {current_route.route}")
                    break
                
                current_route = RouteData([0])
                current_route.node_set = OrderedDictSet([0])
                current_route_cost_upper_bound = 0.0
                route_complete = False
                blocked_nodes = set()
                
                if generate_alternative_first_routes:
                    break
                    
            elif route_interesting:
                log.info(f"Route recorded {current_route.route}")
                routes.append( RouteData(list(current_route.route),
                                         current_route.cost,
                                         current_route.demand,
                                         OrderedDictSet(current_route.node_set)) )
                
        if (sweep_node is not None) and (not routed[sweep_node]):
            current_route.node_set.add(sweep_node)
            routed[sweep_node] = True
            if C:
                current_route.demand+=d[sweep_node]
            if L:
                prev_node = 0
                if len(current_route.route)>2:
                    prev_node = current_route.route[-1] if current_route.route[-1]!=0 else current_route.route[-2]
                ub_delta = -D[prev_node, 0]+D[prev_node, sweep_node]+D[sweep_node,0]
                current_route_cost_upper_bound+=ub_delta
            
            log.debug(f"Added n{sweep_node} to the route set")
        
        log.debug(f"Step to a next sweep node from position {sweep_pos} (n{sweep_pos_to_node_idx(sweep_pos)}) with {list(blocked_nodes)} blocked.")
        
        start_stepping_from = sweep_pos
        while True:
            sweep_pos = _step(sweep_pos, step_inc, max_sweep_idx)
            sweep_node = sweep_pos_to_node_idx(sweep_pos)
            
            if (not routed[sweep_node]) and (sweep_node not in blocked_nodes):
                break
                
            if sweep_pos == start_stepping_from:
                sweep_node = None
                route_complete = True
                blocked_nodes = set()
                break
        
    if inter_route_callback:
        routes = inter_route_callback(routes, callback_data)
        
    return routes

def get_sweep_from_polar_coordinates(rhos,phis):
    N = len(rhos)
    customer_phirhos = np.stack( (phis[1:],rhos[1:],np.arange(1,N)) )
    sweep_node_order = np.argsort(customer_phirhos[0])
    sweep = customer_phirhos[:,sweep_node_order]
    return sweep

def get_sweep_from_cartesian_coordinates(pts):
    """Convert cartesian coordinates of the customer to polar coordinates
    centered at the depot. Returns a sweep which is a stack of 3 vectors:
        
     * phi angles for all customers 
     * rho distances all customers 
     * customer indexes (as float)
    
    Also the phi angles are returned (including the depot)
    Note that the sweep is a size of N-1, as it does not include the depot.
    """
    
    np_pts = pts if isinstance(pts, np.ndarray) else np.asarray(pts)
    depot_x, depot_y = pts[0]
    rhos, phis = cart2pol(np_pts[:,0]-depot_x, np_pts[:,1]-depot_y)
    return get_sweep_from_polar_coordinates(rhos,phis)

def sweep_init(coordinates, D, d, C, L=None, minimize_K=False,
               direction="both", seed_node=BEST_ALTERNATIVE,
               routing_algo=None, **callbacks):
    """
    This algorithm was proposed in Wren (1971) and in Wren & Holliday
    (1972). Sweep was also proposed in Gillett and Miller (1974) who
    gave the algorithm its name. The proposed variants differ in on how many
    starting locations (seed) for the sweep are considered: four in Wren &
    Holliday (1972) and all possible in both directions in Gillett and Miller
    (1974). Also, the improvement procedures differ. The version in this file
    is basebones as as it does not include any route improvement heuristics.
    For implementations of Gillett and Miller (1974) or  Wren & Holliday (1972)
    algorithms, please see their Python files (gillet_miller_sweep.py and
    wren_holliday_sweep.py).
    
    The basic principle of the Sweep algorithm is simple: The algorithm assumes
    that the distances of the CVRP are symmetric, and, furthermore, that the
    points are located on a plane. The catresian coordinates of these points
    in relation to the depot are converted to polar coordinates (rho, phi), and
    then sorted by phi. Starting from an arbitary node (in this implementation
    the default is the one closest to the depot) create a new route and add 
    next  adjecent unrouted node according to their angular coordinate. Repeat 
    as long as the capacity is not exceeded. When this happens, start a new 
    route and repeat the procedure until all nodes are routed. Finally, the 
    routes can optionally be optimized using a TSP algorithm. 
       
    Note that the algorithm gives different results depending on the direction 
    the nodes are inserted. The direction parameter can be "cw" for clockwise 
    insertion order and "ccw" for counterclockwise. As the algorithm is quite
    fast, it is recommended to run it in both directions.
    
    Please note that the actual implementation of the sweep procedure is in the
     do_one_sweep function.
    
    * coordinates can be either 
        a) a list/array of cartesian coordinates (x,y)
        b) a lists/arrays (3) of polar coodinates WITH node indexes (i.e. a 
            numpy stack of phi,rho,idx)
    * D is a numpy ndarray (or equvalent) of the full 2D distance matrix.
    * d is a list of demands. d[0] should be 0.0 as it is the depot.
    * C is the capacity constraint limit for the identical vehicles.
    * L is the optional constraint for the maximum route length/cost/duration.
    * direction is either "cw" or "ccw" depending on the direction the nodes
       are to be processed
    * seed_node is optional parameter that specifies how the first node of the 
       sweep is determned. This can be one of CLOSEST_TO_DEPOT (0),
       SMALLEST_ANGLE (-1), BEST_ALTERNATIVE (-2), which tries every possible 
       staring id, or a positive integer explicitly specifying the node id to
       start from. Also, a list of indexes can be given. These are explicit
       sweep indexes and it is adviseable to give also the sweep parameter.
    
    Wren, A. (1971), "Computers in Transport Planning and Operation", Ian 
      Allan, London.
    Wren, A., and Holliday, A. (1972), "Computer scheduling of vehicles from
      one or more depots to a number of delivery points", Operations Research
      Quarterly 23, 333-344.
    Gillett, B., and Miller, L., (1974). "A heuristic algorithm for the vehicle
      dispatch problem". Operations Research 22, 340-349.
    """
    
    N = len(D)
    if len(coordinates[0])==2:
        sweep = get_sweep_from_cartesian_coordinates(coordinates)
    elif len(coordinates)==3 and (len(coordinates[0])==len(coordinates[1])==len(coordinates[2])):
        sweep = coordinates
    else:
        raise ValueError("The coordinates need to be (x,y) or (phi,rho,node_index,sweep_index_for_node-1). Not "+str(coordinates))
        
    if direction == "ccw":
        step_incs = [1]    
    elif direction == "cw":
        step_incs = [-1]        
    elif direction == "both":
        step_incs = [1,-1]        
    else:
        raise ValueError("""Only "cw", "ccw", and "both" are valid values for the direction parameter""")

    if seed_node==CLOSEST_TO_DEPOT:
        starts = [np.argmin(sweep[1])]
    elif seed_node==SMALLEST_ANGLE:
        starts = [0]
    elif seed_node==BEST_ALTERNATIVE:
        starts = list(range(0,N-1))
    elif type(seed_node) is int:
        starts = [np.where(sweep[2]==abs(seed_node)%N)[0][0]]
    elif type(seed_node) is list:
        starts = seed_node
     
    if routing_algo is None:
        routing_algo = lambda D, node_set: (list(node_set)+[0],
                                            objf(list(node_set)+[0],D))
        
    callback_data = None
    intra_route_callback = None
    inter_route_callback = None
    if 'prepare_callback_datastructures' in callbacks:
        pcds_callback = callbacks['prepare_callback_datastructures']
        callback_data = pcds_callback(D,d,C,L,sweep)
    if 'intra_route_improvement' in callbacks:        
        intra_route_callback = callbacks['intra_route_improvement']
    if 'inter_route_improvement' in callbacks:        
        inter_route_callback = callbacks['inter_route_improvement']
        
    best_sol = None
    best_f = None  
    best_K = None
    
    try:
        for step_inc in step_incs:
            for start in starts:
                log.info(f"\nDo a sweep from position {start} (n{sweep[2][start]}) by steps of {step_inc}")
                
                routes = do_one_sweep(N, D, d, C, L, routing_algo,
                                           sweep, start, step_inc,
                                           False,
                                           intra_route_callback,
                                           inter_route_callback,
                                           callback_data)            
                    
                sol = [n for rd in routes for n in rd.route[:-1]]+[0]
                sol = without_empty_routes(sol)
                sol_f = objf( sol, D )   
                sol_K = sol.count(0)-1
        
                log.info(f"Previous sweep produced solution {sol} ({sol_f:.2f})\n")
                    
                if is_better_sol(best_f, best_K, sol_f, sol_K, minimize_K):
                    best_sol = sol
                    best_f = sol_f
                    best_K = sol_K
    except KeyboardInterrupt:
        raise KeyboardInterrupt(best_sol)
        
    return best_sol

# 简化版本的TSP求解器
def nearest_neighbor_tsp(D, nodes):
    """最近邻TSP求解器"""
    if len(nodes) <= 1:
        return [0, 0], 0.0
    
    unvisited = set(nodes[1:])  # 排除起始点0
    current = nodes[0]
    route = [current]
    total_cost = 0.0
    
    while unvisited:
        # 找到最近未访问节点
        next_node = min(unvisited, key=lambda x: D[current, x])
        total_cost += D[current, next_node]
        route.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    
    # 返回起点
    total_cost += D[current, nodes[0]]
    route.append(nodes[0])
    
    return route, total_cost

def generate_random_cvrp(n_customers=20, depot_location=(0, 0), area_size=100, max_demand=10, vehicle_capacity=50):
    """生成随机CVRP问题"""
    np.random.seed(42)  
    
    # 生成客户坐标
    customer_locations = np.random.rand(n_customers, 2) * area_size
    locations = np.vstack([np.array(depot_location), customer_locations])
    
    # 计算距离矩阵
    n_nodes = n_customers + 1
    D = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            D[i, j] = np.sqrt(np.sum((locations[i] - locations[j])**2))
    
    # 生成需求
    d = [0] + list(np.random.randint(1, max_demand+1, n_customers))
    
    return locations, D, d, vehicle_capacity


def main():
    """主函数：生成随机问题并求解"""
    print("=== 随机CVRP问题求解器 ===")
    
    # 参数设置
    n_customers = 20
    vehicle_capacity = 50
    
    # 生成随机问题
    print(f"\n生成随机CVRP问题...")
    print(f"- 客户数量: {n_customers}")
    print(f"- 车辆容量: {vehicle_capacity}")
    
    locations, D, d, C = generate_random_cvrp(
        n_customers=n_customers, 
        vehicle_capacity=vehicle_capacity
    )
    
    print(f"\n问题信息:")
    print(f"- 总需求: {sum(d)}")
    print(f"- 最小所需车辆数: {sum(d) / C:.1f}")
    
    # 显示客户位置
    print(f"\n客户位置:")
    print(f"仓库: ({locations[0, 0]:.1f}, {locations[0, 1]:.1f})")
    for i in range(1, len(locations)):
        print(f"客户{i}: ({locations[i, 0]:.1f}, {locations[i, 1]:.1f}), 需求: {d[i]}")
    
    # 使用Sweep算法求解
    print(f"\n使用Sweep算法求解...")
    
    # 使用最近邻作为TSP求解器
    routing_algo = nearest_neighbor_tsp
    
    solution = sweep_init(
        coordinates=locations,
        D=D,
        d=d,
        C=C,
        L=None,  # 无路径长度限制
        minimize_K=False,
        direction="both",
        seed_node=BEST_ALTERNATIVE,
        routing_algo=routing_algo
    )
    
    # 分析解决方案
    print(f"\n=== 解决方案 ===")
    print(f"总距离: {objf(solution, D):.2f}")
    
    # 提取各条路径
    routes = []
    current_route = []
    for node in solution:
        if node == 0 and current_route:
            routes.append(current_route + [0])
            current_route = [0]
        else:
            current_route.append(node)
    
    if current_route and len(current_route) > 1:
        routes.append(current_route)
    
    print(f"\n车辆路径:")
    for i, route in enumerate(routes):
        route_demand = sum(d[node] for node in route)
        route_distance = objf(route, D)
        print(f"车辆 {i+1}: {route} (需求: {route_demand}, 距离: {route_distance:.2f})")
    
    return solution, routes, objf(solution, D)

if __name__ == "__main__":
    solution, routes, total_distance = main()
    print(f"\n求解完成！总距离: {total_distance:.2f}")