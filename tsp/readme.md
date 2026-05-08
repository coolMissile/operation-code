# TSP (Traveling Salesman Problem)

## 目录结构

```
tsp/
├── exact/                    # 精确算法（保证最优解）
│   ├── heldkarp.py           # Held-Karp 动态规划
│   ├── Branch_and_Bound.py   # 分支定界
│   └── pyconcorde/           # Concorde solver (Python wrapper)
├── approximation/            # 近似算法（有理论保证）
│   ├── Christofides.py       # Christofides 算法 (3/2-近似)
│   └── greedyMST.py          # 贪心 MST 近似
├── classic/                  # 经典启发式
│   ├── NearestNeighbor.py    # 最近邻
│   ├── NearestNeighbor_simple.py  # 最近邻（简化版）
│   └── NearestInsertion.py   # 最近插入
├── local_search/             # 局部搜索
│   ├── 2_opt_collection.py   # 2-opt 变体
│   ├── lk_heuristic/         # Lin-Kernighan Heuristic (Python)
│   └── LKH3/                 # LKH-3 (Keld Helsgaun, C)
└── meta/                     # 元启发式
    ├── SA.py                 # 模拟退火
    ├── GA.py                 # 遗传算法
    ├── PSO.py                # 粒子群优化
    ├── aco.py                # 蚁群算法
    ├── gls.py                # Guided Local Search
    ├── pomo/                 # POMO (RL-based, PyTorch)
    └── or-tools-ref/         # Google OR-Tools CP-SAT 参考实现
```

## 来源

部分实现来自以下开源项目：
- https://github.com/dmishin/tsp-solver.git
- https://github.com/trevlovett/Python-Ant-Colony-TSP-Solver.git
- https://github.com/kellenf/TSP_collection.git
- https://github.com/yd-kwon/POMO.git
- https://github.com/davidar/LKH3.git
- https://github.com/kikocastroneto/lk_heuristic.git
- https://github.com/digital-brain-sh/pyconcorde.git
- https://github.com/google/or-tools.git
- https://github.com/ppoffice/ant-colony-tsp.git
- https://github.com/chncyhn/simulated-annealing-tsp.git
