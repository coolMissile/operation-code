import random
import numpy as np
import matplotlib.pyplot as plt


class PSO(object):
    def __init__(self, num_city, data):
        self.iter_max = 500          # 迭代次数
        self.num = 200               # 粒子数
        self.num_city = num_city     # 城市数
        self.location = data
        self.dis_mat = self.compute_dis_mat(num_city, self.location)

        # PSO 参数
        self.w = 0.8       # 惯性权重
        self.c1 = 2.0      # 个体学习因子
        self.c2 = 2.0      # 社会学习因子

        # 初始化粒子群
        self.particles = self.init_particles()
        self._eval_fitness()

        # 记录全局最优
        self.gbest_position = self.particles[0]['position'].copy()
        self.gbest_length = self.particles[0]['fitness']

        # 输出解及收敛记录
        self.best_l = self.gbest_length
        self.best_path = self.gbest_position.copy()
        self.iter_x = [0]
        self.iter_y = [self.gbest_length]

    # ─── 初始化 ────────────────────────────────────────────────
    def init_particles(self):
        """混合初始化：前 num_city 个用贪心最近邻，其余随机打乱"""
        particles = []
        # 贪心最近邻初始化
        greedy_paths = self._greedy_paths(min(self.num, self.num_city))
        for path in greedy_paths:
            particles.append({
                'position': path,
                'velocity': self._random_velocity(),
                'pbest_position': path.copy(),
                'pbest_length': float('inf'),
            })
        # 剩余粒子随机初始化
        base = list(range(self.num_city))
        for _ in range(self.num - len(particles)):
            random.shuffle(base)
            p = base.copy()
            particles.append({
                'position': p,
                'velocity': self._random_velocity(),
                'pbest_position': p.copy(),
                'pbest_length': float('inf'),
            })
        return particles

    def _greedy_paths(self, count):
        """为前 count 个城市各生成一条最近邻路径"""
        paths = []
        for start in range(count):
            rest = set(range(self.num_city))
            rest.remove(start)
            path = [start]
            current = start
            while rest:
                nxt = min(rest, key=lambda x: self.dis_mat[current][x])
                path.append(nxt)
                rest.remove(nxt)
                current = nxt
            paths.append(path)
        return paths

    def _random_velocity(self):
        """每条速度是一组交换 (i, j) 的列表"""
        k = np.random.randint(0, self.num_city // 3 + 1)
        v = []
        indices = list(range(self.num_city))
        for _ in range(k):
            i, j = np.random.choice(indices, 2, replace=False)
            v.append((int(i), int(j)))
        return v

    # ─── 适应度 ────────────────────────────────────────────────
    def _eval_fitness(self):
        for p in self.particles:
            length = self._path_length(p['position'])
            p['fitness'] = length
            if length < p['pbest_length']:
                p['pbest_length'] = length
                p['pbest_position'] = p['position'].copy()
            if length < self.gbest_length:
                self.gbest_length = length
                self.gbest_position = p['position'].copy()

    def _path_length(self, path):
        d = self.dis_mat
        total = d[path[-1]][path[0]]
        for i in range(len(path) - 1):
            total += d[path[i]][path[i + 1]]
        return total

    # ─── 核心：Swap-Operator PSO ───────────────────────────────
    def _subtract(self, target, source):
        """生成一组 swap，将 source 变换为 target（基本差运算）"""
        swaps = []
        temp = source.copy()
        for i in range(len(temp)):
            if temp[i] != target[i]:
                j = temp.index(target[i], i)
                temp[i], temp[j] = temp[j], temp[i]
                swaps.append((i, j))
        return swaps

    def _apply_velocity(self, position, velocity):
        """对位置依次施加速度中的所有交换"""
        pos = position.copy()
        for i, j in velocity:
            pos[i], pos[j] = pos[j], pos[i]
        return pos

    def _update_velocity(self, p):
        """V' = w*V  +  c1*r1*(pbest - X)  +  c2*r2*(gbest - X)"""
        # w * V：按惯性权重截断旧速度
        keep_n = int(self.w * len(p['velocity']))
        new_v = p['velocity'][:keep_n]

        # c1 * r1 * (pbest - X)
        swaps_to_pbest = self._subtract(p['pbest_position'], p['position'])
        r1 = np.random.random()
        n1 = int(self.c1 * r1 * len(swaps_to_pbest))
        new_v.extend(swaps_to_pbest[:n1])

        # c2 * r2 * (gbest - X)
        swaps_to_gbest = self._subtract(self.gbest_position, p['position'])
        r2 = np.random.random()
        n2 = int(self.c2 * r2 * len(swaps_to_gbest))
        new_v.extend(swaps_to_gbest[:n2])

        p['velocity'] = new_v

    # ─── 迭代 ──────────────────────────────────────────────────
    def pso(self):
        for cnt in range(1, self.iter_max):
            for p in self.particles:
                # 1. 更新速度
                self._update_velocity(p)
                # 2. 更新位置
                p['position'] = self._apply_velocity(p['position'], p['velocity'])
                # 3. 评价
                length = self._path_length(p['position'])
                p['fitness'] = length
                # 4. 更新 pbest
                if length < p['pbest_length']:
                    p['pbest_length'] = length
                    p['pbest_position'] = p['position'].copy()
                # 5. 更新 gbest
                if length < self.gbest_length:
                    self.gbest_length = length
                    self.gbest_position = p['position'].copy()

            # 记录收敛
            if self.gbest_length < self.best_l:
                self.best_l = self.gbest_length
                self.best_path = self.gbest_position.copy()
            self.iter_x.append(cnt)
            self.iter_y.append(self.best_l)
            print(cnt, self.best_l)

        return self.best_l, self.best_path

    def run(self):
        best_length, best_path = self.pso()
        return self.location[best_path], best_length

    # ─── 距离计算（不变）─────────────────────────────────────────
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


# ─── 读取 TSP 数据 ────────────────────────────────────────────
def read_tsp(path):
    lines = open(path, 'r').readlines()
    assert 'NODE_COORD_SECTION\n' in lines
    index = lines.index('NODE_COORD_SECTION\n')
    data = lines[index + 1:-1]
    tmp = []
    for line in data:
        line = line.strip().split(' ')
        if line[0] == 'EOF':
            continue
        tmpline = []
        for x in line:
            if x == '':
                continue
            else:
                tmpline.append(float(x))
        if tmpline == []:
            continue
        tmp.append(tmpline)
    return tmp


data = read_tsp('data/st70.tsp')

data = np.array(data)
data = data[:, 1:]

model = PSO(num_city=data.shape[0], data=data.copy())
Best_path, Best = model.run()

Best_path = np.vstack([Best_path, Best_path[0]])
fig, axs = plt.subplots(2, 1, sharex=False, sharey=False)
axs[0].scatter(Best_path[:, 0], Best_path[:, 1])
Best_path = np.vstack([Best_path, Best_path[0]])
axs[0].plot(Best_path[:, 0], Best_path[:, 1])
axs[0].set_title('规划结果')
iterations = model.iter_x
best_record = model.iter_y
axs[1].plot(iterations, best_record)
axs[1].set_title('收敛曲线')
plt.show()
