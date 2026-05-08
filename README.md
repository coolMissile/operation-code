# operation-code

组合优化算法的实现集合。

## 目录结构

```
├── tsp/                    # TSP 相关实现
│   ├── exact/              # 精确算法（Concorde, Branch & Bound...）
│   ├── approximation/      # 近似算法（Christofides...）
│   ├── classic/            # 经典启发式（Nearest Neighbor, Insertion...）
│   ├── local_search/       # 局部搜索（2-opt, LK, LKH3...）
│   └── meta/               # 元启发式（GA, SA, ACO, PSO, POMO...）
└── cvrp/                   # CVRP 相关实现
```
