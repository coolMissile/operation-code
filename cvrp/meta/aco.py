import torch
from torch.distributions import Categorical
import numpy as np
import time
from typing import List, Tuple, Optional


class VRPSolver:
    """
    完整的VRP问题求解器
    使用蚁群算法（ACO）解决带容量约束的车辆路径问题
    由MCTS-AHD-master的aco.py改编
    """
    
    def __init__(self, 
                 n_customers: int,
                 capacity: float,
                 n_vehicles: int = None,
                 coordinates: Optional[np.ndarray] = None,
                 distances: Optional[np.ndarray] = None,
                 demands: Optional[np.ndarray] = None,
                 n_ants: int = 30,
                 decay: float = 0.9,
                 alpha: float = 1.0,
                 beta: float = 2.0,
                 device: str = 'cpu'):
        """
        初始化VRP求解器
        
        参数:
        ----------
        n_customers : int
            客户数量（不包括仓库）
        capacity : float
            车辆容量限制
        n_vehicles : int, optional
            车辆数量限制，None表示不限制
        coordinates : np.ndarray, optional
            节点坐标，形状为 (n+1, 2)，第0个是仓库
        distances : np.ndarray, optional
            距离矩阵，形状为 (n+1, n+1)
        demands : np.ndarray, optional
            需求数组，形状为 (n+1,)，仓库需求为0
        n_ants : int
            蚂蚁数量
        decay : float
            信息素挥发率
        alpha : float
            信息素重要性
        beta : float
            启发式信息重要性
        device : str
            计算设备
        """
        self.n_customers = n_customers
        self.n_nodes = n_customers + 1  # 包括仓库
        self.capacity = capacity
        self.n_vehicles = n_vehicles
        self.device = device
        
        # 生成或验证输入数据
        if coordinates is not None:
            self.coordinates = torch.tensor(coordinates, device=device) if not isinstance(coordinates, torch.Tensor) else coordinates
            self.distances = self._calculate_distances(coordinates)
        elif distances is not None:
            self.distances = torch.tensor(distances, device=device) if not isinstance(distances, torch.Tensor) else distances
            self.coordinates = None
        else:
            # 生成随机问题
            self.coordinates, self.distances, self.demands = self.generate_random_problem(n_customers)
            self.coordinates = torch.tensor(self.coordinates, device=device)
            self.distances = torch.tensor(self.distances, device=device)
            self.demands = torch.tensor(self.demands, device=device)
        
        if demands is not None:
            self.demands = torch.tensor(demands, device=device) if not isinstance(demands, torch.Tensor) else demands
        elif not hasattr(self, 'demands'):
            # 生成随机需求
            self.demands = torch.cat([
                torch.tensor([0.0], device=device),  # 仓库需求为0
                torch.rand(n_customers, device=device) * 0.8 + 0.1
            ])
        
        # 初始化启发式信息（这里使用1/距离）
        self.heuristic = 1.0 / (self.distances + 1e-10)
        self.heuristic[self.heuristic > 1e5] = 1e5  # 防止溢出
        
        # 初始化ACO
        self.aco = ACO(
            distances=self.distances,
            demand=self.demands,
            heuristic=self.heuristic,
            capacity=capacity,
            n_ants=n_ants,
            decay=decay,
            alpha=alpha,
            beta=beta,
            device=device
        )
        
        # 记录求解历史
        self.history = {
            'costs': [],
            'solutions': [],
            'times': []
        }
        
    def _calculate_distances(self, coordinates: np.ndarray) -> torch.Tensor:
        """计算欧氏距离矩阵"""
        if isinstance(coordinates, torch.Tensor):
            coords_np = coordinates.cpu().numpy()
        else:
            coords_np = coordinates
            
        n = len(coords_np)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    distances[i][j] = np.linalg.norm(coords_np[i] - coords_np[j])
        
        return torch.tensor(distances, device=self.device)
    
    @staticmethod
    def generate_random_problem(n_customers: int, 
                               coord_range: Tuple[float, float] = (0, 100),
                               demand_range: Tuple[float, float] = (1, 10)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """生成随机VRP问题实例"""
        # 生成坐标
        coordinates = np.random.rand(n_customers + 1, 2) * (coord_range[1] - coord_range[0]) + coord_range[0]
        
        # 将第一个点设为仓库
        coordinates[0] = np.array([coord_range[0] + (coord_range[1] - coord_range[0]) / 2, 
                                  coord_range[0] + (coord_range[1] - coord_range[0]) / 2])
        
        # 计算距离矩阵
        distances = np.zeros((n_customers + 1, n_customers + 1))
        for i in range(n_customers + 1):
            for j in range(n_customers + 1):
                if i != j:
                    distances[i][j] = np.linalg.norm(coordinates[i] - coordinates[j])
        
        # 生成需求
        demands = np.zeros(n_customers + 1)
        demands[1:] = np.random.rand(n_customers) * (demand_range[1] - demand_range[0]) + demand_range[0]
        
        return coordinates, distances, demands
    
    def solve(self, n_iterations: int = 100, verbose: bool = True) -> dict:
        """
        求解VRP问题
        
        参数:
        ----------
        n_iterations : int
            迭代次数
        verbose : bool
            是否显示进度信息
            
        返回:
        ----------
        dict: 包含解决方案信息的字典
        """
        if verbose:
            print(f"开始求解VRP问题...")
            print(f"客户数量: {self.n_customers}")
            print(f"车辆容量: {self.capacity}")
            print(f"总需求: {self.demands[1:].sum().item():.2f}")
            print(f"迭代次数: {n_iterations}")
            print("-" * 50)
        
        start_time = time.time()
        
        # 运行ACO算法
        best_cost = self.aco.run(n_iterations)
        
        # 提取解决方案
        solution = self.extract_solution(self.aco.shortest_path)
        
        end_time = time.time()
        
        # 记录历史
        self.history['costs'].append(best_cost.item())
        self.history['solutions'].append(solution)
        self.history['times'].append(end_time - start_time)
        
        if verbose:
            print(f"求解完成!")
            print(f"最优成本: {best_cost:.2f}")
            print(f"计算时间: {end_time - start_time:.2f}秒")
            print(f"车辆使用数量: {solution['n_vehicles']}")
            print(f"平均每辆车服务客户数: {solution['avg_customers_per_vehicle']:.1f}")
            print(f"平均车辆使用率: {solution['avg_utilization'] * 100:.1f}%")
            print("-" * 50)
            
            # 打印路径详情
            for i, route in enumerate(solution['routes']):
                route_demand = sum(self.demands[node].item() for node in route if node != 0)
                route_length = self._calculate_route_length(route)
                print(f"车辆{i+1}: 路径 {route} | 需求 {route_demand:.1f}/{self.capacity} | 距离 {route_length:.2f}")
        
        return solution
    
    def extract_solution(self, path: torch.Tensor) -> dict:
        """从路径中提取解决方案"""
        path = path.cpu().numpy() if torch.is_tensor(path) else path
        routes = []
        current_route = []
        
        for node in path:
            if node == 0:  # 仓库
                if current_route:  # 如果不是空路径
                    routes.append(current_route + [0])  # 添加返回仓库
                    current_route = []
            else:
                current_route.append(int(node))
        
        # 如果有未完成的路径
        if current_route:
            routes.append(current_route + [0])
        
        # 计算统计信息
        total_cost = self.aco.lowest_cost.item()
        n_vehicles = len(routes)
        
        # 计算每辆车的利用率
        utilizations = []
        for route in routes:
            route_demand = sum(self.demands[node].item() for node in route if node != 0)
            utilizations.append(route_demand / self.capacity)
        
        avg_utilization = np.mean(utilizations) if utilizations else 0
        
        return {
            'path': path,
            'routes': routes,
            'cost': total_cost,
            'n_vehicles': n_vehicles,
            'avg_utilization': avg_utilization,
            'avg_customers_per_vehicle': self.n_customers / n_vehicles if n_vehicles > 0 else 0
        }
    
    def _calculate_route_length(self, route: List[int]) -> float:
        """计算单条路径的长度"""
        length = 0.0
        for i in range(len(route) - 1):
            length += self.distances[route[i], route[i+1]].item()
        return length
    
    def get_solution_details(self, solution: dict = None) -> str:
        """获取解决方案的详细文本描述"""
        if solution is None and hasattr(self.aco, 'shortest_path'):
            solution = self.extract_solution(self.aco.shortest_path)
        
        output_lines = []
        output_lines.append("=" * 60)
        output_lines.append("VRP解决方案详情")
        output_lines.append("=" * 60)
        output_lines.append(f"总成本: {solution['cost']:.2f}")
        output_lines.append(f"车辆使用数量: {solution['n_vehicles']}")
        output_lines.append(f"平均车辆使用率: {solution['avg_utilization'] * 100:.1f}%")
        output_lines.append(f"平均每辆车服务客户数: {solution['avg_customers_per_vehicle']:.1f}")
        output_lines.append("-" * 60)
        
        for i, route in enumerate(solution['routes']):
            route_demand = sum(self.demands[node].item() for node in route if node != 0)
            route_length = self._calculate_route_length(route)
            output_lines.append(f"车辆{i+1}:")
            output_lines.append(f"  路径: {' -> '.join(map(str, route))}")
            output_lines.append(f"  总需求: {route_demand:.1f}/{self.capacity} ({route_demand/self.capacity*100:.1f}%)")
            output_lines.append(f"  路径长度: {route_length:.2f}")
            output_lines.append(f"  服务客户: {[node for node in route if node != 0]}")
            output_lines.append("-" * 40)
        
        return "\n".join(output_lines)
    
    def get_convergence_info(self) -> str:
        """获取收敛信息"""
        if not self.history['costs']:
            return "没有历史数据"
        
        best_cost = min(self.history['costs'])
        iterations = len(self.history['costs'])
        avg_time = np.mean(self.history['times']) if self.history['times'] else 0
        
        info_lines = []
        info_lines.append("=" * 60)
        info_lines.append("收敛信息")
        info_lines.append("=" * 60)
        info_lines.append(f"迭代次数: {iterations}")
        info_lines.append(f"历史最优成本: {best_cost:.2f}")
        info_lines.append(f"平均求解时间: {avg_time:.3f}秒")
        info_lines.append(f"总求解时间: {sum(self.history['times']):.3f}秒")
        
        if len(self.history['costs']) > 1:
            improvement = (self.history['costs'][0] - best_cost) / self.history['costs'][0] * 100
            info_lines.append(f"相对改进: {improvement:.2f}%")
        
        return "\n".join(info_lines)


# ACO算法类保持不变
class ACO():
    """
    蚁群优化算法实现
    """
    
    def __init__(self,  # 0: depot
                 distances,  # (n, n)
                 demand,    # (n, )
                 heuristic,  # (n, n)
                 capacity,
                 n_ants=30, 
                 decay=0.9,
                 alpha=1,
                 beta=1,
                 device='cpu'):
        
        self.problem_size = len(distances)
        self.distances = torch.tensor(distances, device=device) if not isinstance(distances, torch.Tensor) else distances
        self.demand = torch.tensor(demand, device=device) if not isinstance(demand, torch.Tensor) else demand
        self.capacity = capacity
                
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        
        self.pheromone = torch.ones_like(self.distances)
        self.heuristic = torch.tensor(heuristic, device=device) if not isinstance(heuristic, torch.Tensor) else heuristic

        self.shortest_path = None
        self.lowest_cost = float('inf')

        self.device = device
        self.history = {'costs': [], 'best_costs': []}
    
    @torch.no_grad()
    def run(self, n_iterations):
        for iteration in range(n_iterations):
            paths = self.gen_path()
            costs = self.gen_path_costs(paths)
            
            best_cost, best_idx = costs.min(dim=0)
            if best_cost < self.lowest_cost:
                self.shortest_path = paths[:, best_idx]
                self.lowest_cost = best_cost
            
            # 记录历史
            self.history['costs'].append(costs.cpu().numpy())
            self.history['best_costs'].append(self.lowest_cost.item())
            
            if iteration % 10 == 0:
                print(f"迭代 {iteration}: 当前最优成本 = {self.lowest_cost:.2f}")
            
            self.update_pheronome(paths, costs)

        return self.lowest_cost
       
    @torch.no_grad()
    def update_pheronome(self, paths, costs):
        '''
        Args:
            paths: torch tensor with shape (problem_size, n_ants)
            costs: torch tensor with shape (n_ants,)
        '''
        self.pheromone = self.pheromone * self.decay 
        for i in range(self.n_ants):
            path = paths[:, i]
            cost = costs[i]
            self.pheromone[path[:-1], torch.roll(path, shifts=-1)[:-1]] += 1.0/cost
        self.pheromone[self.pheromone < 1e-10] = 1e-10
    
    @torch.no_grad()
    def gen_path_costs(self, paths):
        u = paths.permute(1, 0)  # shape: (n_ants, max_seq_len)
        v = torch.roll(u, shifts=-1, dims=1)  
        return torch.sum(self.distances[u[:, :-1], v[:, :-1]], dim=1)

    def gen_path(self):
        actions = torch.zeros((self.n_ants,), dtype=torch.long, device=self.device)
        visit_mask = torch.ones(size=(self.n_ants, self.problem_size), device=self.device)
        visit_mask = self.update_visit_mask(visit_mask, actions)
        used_capacity = torch.zeros(size=(self.n_ants,), device=self.device)
        
        used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity)
        
        paths_list = [actions]  # paths_list[i] is the ith move (tensor) for all ants
        
        done = self.check_done(visit_mask, actions)
        while not done:
            actions = self.pick_move(actions, visit_mask, capacity_mask)
            paths_list.append(actions)
            visit_mask = self.update_visit_mask(visit_mask, actions)
            used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity)
            done = self.check_done(visit_mask, actions)
            
        return torch.stack(paths_list)
        
    def pick_move(self, prev, visit_mask, capacity_mask):
        pheromone = self.pheromone[prev]  # shape: (n_ants, p_size)
        heuristic = self.heuristic[prev]  # shape: (n_ants, p_size)
        dist = ((pheromone ** self.alpha) * (heuristic ** self.beta) * visit_mask * capacity_mask)  # shape: (n_ants, p_size)
        dist = Categorical(dist)
        actions = dist.sample()  # shape: (n_ants,)
        return actions
    
    def update_visit_mask(self, visit_mask, actions):
        visit_mask[torch.arange(self.n_ants, device=self.device), actions] = 0
        visit_mask[:, 0] = 1  # depot can be revisited with one exception
        visit_mask[(actions==0) * (visit_mask[:, 1:]!=0).any(dim=1), 0] = 0  # one exception is here
        return visit_mask
    
    def update_capacity_mask(self, cur_nodes, used_capacity):
        '''
        Args:
            cur_nodes: shape (n_ants, )
            used_capacity: shape (n_ants, )
            capacity_mask: shape (n_ants, p_size)
        Returns:
            ant_capacity: updated capacity
            capacity_mask: updated mask
        '''
        capacity_mask = torch.ones(size=(self.n_ants, self.problem_size), device=self.device)
        # update capacity
        used_capacity[cur_nodes==0] = 0
        used_capacity = used_capacity + self.demand[cur_nodes]
        # update capacity_mask
        remaining_capacity = self.capacity - used_capacity  # (n_ants,)
        remaining_capacity_repeat = remaining_capacity.unsqueeze(-1).repeat(1, self.problem_size)  # (n_ants, p_size)
        demand_repeat = self.demand.unsqueeze(0).repeat(self.n_ants, 1)  # (n_ants, p_size)
        capacity_mask[demand_repeat > remaining_capacity_repeat] = 0
        
        return used_capacity, capacity_mask
    
    def check_done(self, visit_mask, actions):
        return (visit_mask[:, 1:] == 0).all() and (actions == 0).all()


#使用示例
def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 示例1: 生成随机问题
    print("\n1. 生成随机CVRP问题并求解")
    print("-" * 40)
    
    # 创建求解器
    solver = VRPSolver(
        n_customers=20,
        capacity=15.0,
        n_ants=50,
        decay=0.95,
        alpha=1.0,
        beta=3.0,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 求解问题
    solution = solver.solve(n_iterations=100, verbose=True)
    
    # 打印详细解决方案
    print(solver.get_solution_details())
    print(solver.get_convergence_info())
    
    print("\n" + "=" * 60)
    print("2. 使用自定义数据求解")
    print("-" * 40)
    
    # 示例2: 使用自定义数据
    n_nodes = 8  # 7个客户 + 1个仓库
    
    # 自定义坐标
    coordinates = np.array([
        [50, 50],   # 仓库
        [20, 20],   # 客户1
        [30, 40],   # 客户2
        [50, 30],   # 客户3
        [60, 60],   # 客户4
        [70, 40],   # 客户5
        [80, 20],   # 客户6
        [40, 80]    # 客户7
    ])
    
    # 自定义需求
    demands = np.array([0, 3, 5, 2, 4, 3, 6, 2])  # 仓库需求为0
    
    # 创建求解器
    custom_solver = VRPSolver(
        n_customers=n_nodes-1,
        capacity=10.0,
        coordinates=coordinates,
        demands=demands,
        n_ants=30,
        decay=0.9,
        alpha=1.0,
        beta=2.0
    )
    
    # 求解
    custom_solution = custom_solver.solve(n_iterations=50, verbose=True)
    
    # 打印详细解决方案
    print(custom_solver.get_solution_details())
    print(custom_solver.get_convergence_info())
    
    print("\n" + "=" * 60)
    print("3. 批量测试不同规模的问题")
    print("-" * 40)
    
    # 示例3: 批量测试
    problem_sizes = [10, 20, 30]
    capacities = [20.0, 30.0, 40.0]
    
    for size, cap in zip(problem_sizes, capacities):
        print(f"\n测试 {size} 个客户，容量 {cap} 的问题:")
        test_solver = VRPSolver(
            n_customers=size,
            capacity=cap,
            n_ants=30,
            decay=0.9,
            alpha=1.0,
            beta=2.0
        )
        
        solution = test_solver.solve(n_iterations=50, verbose=False)
        print(f"  最优成本: {solution['cost']:.2f}")
        print(f"  车辆数量: {solution['n_vehicles']}")
        print(f"  求解时间: {test_solver.history['times'][-1]:.2f}秒")


if __name__ == "__main__":
    main()