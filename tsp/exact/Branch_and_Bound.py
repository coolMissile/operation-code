import math
import functools
from queue import PriorityQueue
import matplotlib.pyplot as plt
import numpy as np

@functools.total_ordering
class Node(object):
    def __init__(self, level=None, path=None, bound=None):
        self.level = level
        self.path = path
        self.bound = bound

    def __eq__(self, other):  # operator ==
        if not isinstance(other, Node):
            return NotImplemented
        return self.bound == other.bound
    
    def __lt__(self, other):  # operator <
        if not isinstance(other, Node):
            return NotImplemented
        return self.bound < other.bound
    
    def __str__(self):
        return str(tuple([self.level, self.path, self.bound]))

class DP(object):
    def __init__(self, num_city, num_total, iteration, data):
        self.num_city = num_city
        self.location = data
        self.dis_mat = self.compute_dis_mat(num_city, data)

    def compute_dis_mat(self, num_city, location):
        dis_mat = np.zeros((num_city, num_city))
        for i in range(num_city):
            for j in range(num_city):
                if i == j:
                    dis_mat[i][j] = np.inf
                    continue
                a = location[i]
                b = location[j]
                tmp = np.sqrt(sum([(x[0] - x[1]) ** 2 for x in zip(a, b)]))
                dis_mat[i][j] = tmp
        return dis_mat


    def run(self, src=0):
        optimal_tour = []
        n = len(self.dis_mat)
        if not n:
            raise ValueError("Invalid adj Matrix")
        u = Node()
        PQ = PriorityQueue()
        optimal_length = 0
        v = Node(level=0, path=[0])
        min_length = float('inf')  # infinity
        v.bound = self.bound(self.dis_mat, v)
        PQ.put(v)
        
        while not PQ.empty():
            print(f"Queue size: {PQ.qsize()}")
            v = PQ.get()
            if v.bound < min_length:
                u.level = v.level + 1
                for i in [x for x in range(1, n) if x not in v.path]:
                    u.path = v.path[:]
                    u.path.append(i)
                    if u.level == n - 2:
                        # 使用列表推导式替代filter
                        l = [x for x in range(1, n) if x not in u.path]
                        u.path.append(l[0])
                        # putting the first vertex at last
                        u.path.append(0)

                        _len = self.length(self.dis_mat, u)
                        if _len < min_length:
                            min_length = _len
                            optimal_length = _len
                            optimal_tour = u.path[:]

                    else:
                        u.bound = self.bound(self.dis_mat, u)
                        if u.bound < min_length:
                            PQ.put(u)
                    # make a new node at each iteration! python it is!!
                    u = Node(level=u.level)

        # shifting to proper source(start of path)
        optimal_tour_src = optimal_tour
        if src != 0:  
            optimal_tour_src = optimal_tour[:-1]
            y = optimal_tour_src.index(src)
            optimal_tour_src = optimal_tour_src[y:] + optimal_tour_src[:y]
            optimal_tour_src.append(optimal_tour_src[0])

        return optimal_tour_src, optimal_length

    def length(self, adj_mat, node):
        tour = node.path
        # returns the sum of two consecutive elements of tour in adj[i][j]
        return sum([adj_mat[tour[i]][tour[i + 1]] for i in range(len(tour) - 1)])

    def bound(self, adj_mat, node):
        path = node.path
        _bound = 0

        n = len(adj_mat)
        determined, last = path[:-1], path[-1]
        # remain is index based
        remain = [x for x in range(n) if x not in path]  

        # for the edges that are certain
        for i in range(len(path) - 1):
            _bound += adj_mat[path[i]][path[i + 1]]

        # for the last item
        if remain:  
            _bound += min([adj_mat[last][i] for i in remain])

        p = [path[0]] + remain
        # for the undetermined nodes
        for r in remain:
            other_nodes = [x for x in p if x != r]
            if other_nodes:  #
                _bound += min([adj_mat[r][i] for i in other_nodes])
        return _bound



if __name__ == "__main__":
    num_city = 5
    num_total = 100
    iteration = 1000
    data = np.random.rand(num_city, 2) * 100
   
    dp = DP(num_city, num_total, iteration, data)
    
    Best_path, Best = dp.run(src=0)
    
    
    print('TSP分支定界算法结果')
    print(f'最佳路径长度: {Best:.2f}')
    print(f'最佳路径: {Best_path}')
    
    
    
