#tsp_collection/GA.py 的改进版本
#选择、交叉、变异各有三种可选操作
import numpy as np
import random
from typing import List, Tuple, Callable


class GeneticAlgorithmTSP:
    def __init__(self, 
                 distance_matrix: np.ndarray,
                 population_size: int = 100,
                 max_generations: int = 1000,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 elitism_size: int = 2,
                 selection_method: str = 'tournament',
                 crossover_method: str = 'ox',
                 mutation_method: str = 'swap'):
        """
        初始化遗传算法
        
        Args:
            distance_matrix: 距离矩阵
            population_size: 种群大小
            max_generations: 最大迭代次数
            crossover_rate: 交叉概率
            mutation_rate: 变异概率
            elitism_size: 精英保留数量
            selection_method: 选择方法 ('tournament', 'roulette', 'rank')
            crossover_method: 交叉方法 ('ox', 'cx', 'pmx')
            mutation_method: 变异方法 ('swap', 'inversion', 'scramble')
        """
        self.distance_matrix = distance_matrix
        self.n_cities = len(distance_matrix)
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_size = elitism_size
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        
        # 初始化种群
        self.population = self._initialize_population()
        self.best_solution = None
        self.best_fitness = float('inf')
        #self.fitness_history = []
        
    def _initialize_population(self) -> List[List[int]]:
        population = []
        
        # 1/4 使用最近邻启发式
        for _ in range(self.population_size // 4):
            population.append(self._nearest_neighbor())
        
        # 1/4 使用最近插入
        for _ in range(self.population_size // 4):
            population.append(self._nearest_insertion())
        
        # 剩下的随机生成
        remaining = self.population_size - len(population)
        for _ in range(remaining):
            ind = list(range(self.n_cities))
            random.shuffle(ind)
            population.append(ind)
        
        return population
    
    def _nearest_neighbor(self) -> List[int]:
        """最近邻构造初始解"""
        start = random.randint(0, self.n_cities - 1)
        unvisited = set(range(self.n_cities))
        unvisited.remove(start)
        tour = [start]
        current = start
        
        while unvisited:
            # 找到最近的城市
            nearest = min(unvisited, 
                        key=lambda x: self.distance_matrix[current][x])
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        return tour
    
    def _nearest_insertion(self) -> List[int]:
        """最近插入构造初始解"""
        # 随机选择3个城市作为初始子回路
        cities = list(range(self.n_cities))
        random.shuffle(cities)
        tour = cities[:3]
        unvisited = set(cities[3:])
        
        while unvisited:
            # 找到离当前回路最近的城市
            min_distance = float('inf')
            city_to_insert = -1
            
            for city in unvisited:
                for city_in_tour in tour:
                    dist = self.distance_matrix[city][city_in_tour]
                    if dist < min_distance:
                        min_distance = dist
                        city_to_insert = city
            
            # 找到最佳插入位置
            best_position = -1
            best_increase = float('inf')
            
            for i in range(len(tour)):
                j = (i + 1) % len(tour)
                increase = (self.distance_matrix[tour[i]][city_to_insert] +
                          self.distance_matrix[city_to_insert][tour[j]] -
                          self.distance_matrix[tour[i]][tour[j]])
                
                if increase < best_increase:
                    best_increase = increase
                    best_position = i + 1
            
            # 插入城市
            tour.insert(best_position, city_to_insert)
            unvisited.remove(city_to_insert)
        
        return tour
    
    def _calculate_fitness(self, individual: List[int]) -> float:
        """计算适应度（路径长度）"""
        total_distance = 0
        n = len(individual)
        
        for i in range(n):
            j = (i + 1) % n
            total_distance += self.distance_matrix[individual[i]][individual[j]]
        
        return total_distance
    
    def _tournament_selection(self, tournament_size: int = 3) -> Tuple[List[int], List[int]]:
        """锦标赛选择"""
        # 选择父代1
        tournament1 = random.sample(self.population, tournament_size)
        parent1 = min(tournament1, key=self._calculate_fitness)
        
        # 选择父代2（确保不同）
        tournament2 = random.sample(self.population, tournament_size)
        parent2 = min(tournament2, key=self._calculate_fitness)
        
        return parent1, parent2
    
    def _roulette_wheel_selection(self) -> Tuple[List[int], List[int]]:
        """轮盘赌选择"""
        # 计算适应度
        fitness_values = [self._calculate_fitness(ind) for ind in self.population]
        max_fitness = max(fitness_values)
        
        # 转换为选择概率（距离越小，概率越大）
        selection_probs = [(max_fitness - fit + 1) for fit in fitness_values]
        total = sum(selection_probs)
        selection_probs = [prob / total for prob in selection_probs]
        
        # 选择
        idx1 = np.random.choice(len(self.population), p=selection_probs)
        idx2 = np.random.choice(len(self.population), p=selection_probs)
        
        return self.population[idx1], self.population[idx2]
    
    def _rank_selection(self) -> Tuple[List[int], List[int]]:
        """排序选择"""
        # 按适应度排序
        sorted_pop = sorted(self.population, key=self._calculate_fitness)
        n = len(sorted_pop)
        
        # 分配选择概率
        probs = [2*(n - i) / (n*(n+1)) for i in range(n)]
        
        # 选择
        idx1 = np.random.choice(n, p=probs)
        idx2 = np.random.choice(n, p=probs)
        
        return sorted_pop[idx1], sorted_pop[idx2]
    
    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """顺序交叉（Order Crossover, OX）"""
        size = len(parent1)
        
        # 随机选择两个切点
        point1 = random.randint(0, size - 2)
        point2 = random.randint(point1 + 1, size - 1)
        
        # 初始化子代
        child1 = [-1] * size
        child2 = [-1] * size
        
        # 复制中间段
        child1[point1:point2+1] = parent1[point1:point2+1]
        child2[point1:point2+1] = parent2[point1:point2+1]
        
        # 填充剩余位置
        self._fill_child_ox(child1, parent2, point1, point2)
        self._fill_child_ox(child2, parent1, point1, point2)
        
        return child1, child2
    
    def _fill_child_ox(self, child: List[int], parent: List[int], start: int, end: int):
        """为OX交叉填充子代"""
        size = len(child)
        parent_pos = 0
        
        for i in range(size):
            pos = (end + 1 + i) % size
            
            if pos < start or pos > end:
                while parent[parent_pos % size] in child:
                    parent_pos += 1
                child[pos] = parent[parent_pos % size]
                parent_pos += 1
    
    def _cycle_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """循环交叉（Cycle Crossover, CX）"""
        size = len(parent1)
        child1 = [-1] * size
        child2 = [-1] * size
        
        # 找循环
        cycles = []
        visited = [False] * size
        
        for i in range(size):
            if not visited[i]:
                cycle = []
                current = i
                while not visited[current]:
                    visited[current] = True
                    cycle.append(current)
                    value = parent1[current]
                    current = parent2.index(value)
                cycles.append(cycle)
        
        # 根据循环构建子代
        for i, cycle in enumerate(cycles):
            for idx in cycle:
                if i % 2 == 0:  # 偶数循环
                    child1[idx] = parent1[idx]
                    child2[idx] = parent2[idx]
                else:  # 奇数循环
                    child1[idx] = parent2[idx]
                    child2[idx] = parent1[idx]
        
        return child1, child2
    
    def _partially_mapped_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """部分映射交叉（Partially Mapped Crossover, PMX）"""
        size = len(parent1)
        
        # 随机选择两个切点
        point1 = random.randint(0, size - 2)
        point2 = random.randint(point1 + 1, size - 1)
        
        # 初始化子代
        child1 = [-1] * size
        child2 = [-1] * size
        
        # 复制中间段
        child1[point1:point2+1] = parent1[point1:point2+1]
        child2[point1:point2+1] = parent2[point1:point2+1]
        
        # 创建映射关系
        mapping1 = {}
        mapping2 = {}
        
        for i in range(point1, point2+1):
            mapping1[parent2[i]] = parent1[i]
            mapping2[parent1[i]] = parent2[i]
        
        # 填充剩余位置
        for i in range(size):
            if i < point1 or i > point2:
                val1 = parent2[i]
                val2 = parent1[i]
                
                # 处理冲突
                while val1 in child1[point1:point2+1]:
                    val1 = mapping1.get(val1, val1)
                child1[i] = val1
                
                while val2 in child2[point1:point2+1]:
                    val2 = mapping2.get(val2, val2)
                child2[i] = val2
        
        return child1, child2
    
    def _mutation(self, individual: List[int]) -> List[int]:
        """变异操作"""
        if random.random() > self.mutation_rate:
            return individual
        
        mutant = individual.copy()
        
        if self.mutation_method == 'swap':
            i, j = random.sample(range(len(mutant)), 2)
            mutant[i], mutant[j] = mutant[j], mutant[i]
            
        elif self.mutation_method == 'inversion':
            i, j = sorted(random.sample(range(len(mutant)), 2))
            mutant[i:j+1] = mutant[i:j+1][::-1]
            
        elif self.mutation_method == 'scramble':
            i, j = sorted(random.sample(range(len(mutant)), 2))
            segment = mutant[i:j+1]
            random.shuffle(segment)
            mutant[i:j+1] = segment
        
        return mutant
    
    def _apply_local_search(self, individual: List[int]) -> List[int]:
        """应用2-opt局部搜索"""
        improved = True
        best_distance = self._calculate_fitness(individual)
        current = individual.copy()
        
        while improved:
            improved = False
            for i in range(len(current) - 1):
                for j in range(i + 2, len(current)):
                    if j == len(current) - 1 and i == 0:
                        continue
                    
                    # 尝试2-opt交换
                    new_route = current[:]
                    new_route[i+1:j+1] = current[i+1:j+1][::-1]
                    new_distance = self._calculate_fitness(new_route)
                    
                    if new_distance < best_distance:
                        current = new_route
                        best_distance = new_distance
                        improved = True
                        break
                if improved:
                    break
        
        return current
    
    def run(self) -> Tuple[List[int], float, List[float]]:
        """运行遗传算法"""
        print("开始遗传算法优化...")
        
        for generation in range(self.max_generations):
            # 计算适应度
            fitness_values = [self._calculate_fitness(ind) for ind in self.population]
            
            # 更新最优解
            current_best_idx = np.argmin(fitness_values)
            current_best_fitness = fitness_values[current_best_idx]
            
            if current_best_fitness < self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_solution = self.population[current_best_idx].copy()
            
            #self.fitness_history.append(self.best_fitness)
            
            # 精英保留
            sorted_indices = np.argsort(fitness_values)
            elites = [self.population[i] for i in sorted_indices[:self.elitism_size]]
            
            # 生成新一代
            new_population = elites.copy()
            
            while len(new_population) < self.population_size:
                # 选择
                if self.selection_method == 'tournament':
                    parent1, parent2 = self._tournament_selection()
                elif self.selection_method == 'roulette':
                    parent1, parent2 = self._roulette_wheel_selection()
                else:  # rank
                    parent1, parent2 = self._rank_selection()
                
                # 交叉
                if random.random() < self.crossover_rate:
                    if self.crossover_method == 'ox':
                        child1, child2 = self._order_crossover(parent1, parent2)
                    elif self.crossover_method == 'cx':
                        child1, child2 = self._cycle_crossover(parent1, parent2)
                    else:  # pmx
                        child1, child2 = self._partially_mapped_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # 变异
                child1 = self._mutation(child1)
                child2 = self._mutation(child2)
                
                # 局部搜索（可选）
                if generation % 10 == 0:  # 每10代进行一次局部搜索
                    if random.random() < 0.1:
                        child1 = self._apply_local_search(child1)
                    if random.random() < 0.1:
                        child2 = self._apply_local_search(child2)
                
                new_population.extend([child1, child2])
            
            # 确保种群大小不变
            self.population = new_population[:self.population_size]
            
            # 输出进度
            if generation % 100 == 0:
                print(f"第 {generation} 代, 最优距离: {self.best_fitness:.2f}")
        
        print(f"优化完成！最优距离: {self.best_fitness:.2f}")
        return self.best_solution, self.best_fitness
    
if __name__ == "__main__":
    # 示例使用
    num_cities = 20
    distance_matrix = np.random.rand(num_cities, num_cities) * 100
    distance_matrix = (distance_matrix + distance_matrix.T) / 2  
    np.fill_diagonal(distance_matrix, 0)  
    
    ga_tsp = GeneticAlgorithmTSP(distance_matrix, population_size=200, max_generations=100)
    best_solution, best_fitness = ga_tsp.run()
    
    print("最佳路径:", best_solution)
    print("最佳距离:", best_fitness)