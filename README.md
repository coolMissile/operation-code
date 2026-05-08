# operation-code

组合优化算法的代码收集与演化框架。

## 目录结构

```
├── tsp/                    # TSP 相关实现
│   ├── exact/              # 精确算法（Concorde, Branch & Bound...）
│   ├── approximation/      # 近似算法（Christofides...）
│   ├── classic/            # 经典启发式（Nearest Neighbor, Insertion...）
│   ├── local_search/       # 局部搜索（2-opt, LK, LKH3...）
│   └── meta/               # 元启发式（GA, SA, ACO, PSO, POMO...）
├── cvrp/                   # CVRP 相关实现
├── src/                    # CodeEvolution-CO 框架源码
│   ├── core.py             # 核心数据抽象
│   ├── io/                 # 输入输出（读问题、写方案、读代码）
│   ├── abstraction/        # 算法提取与抽象（LLM 分析→设计空间）
│   ├── evolution/          # 演化引擎（生成→评估→闭环）
│   ├── evaluation/         # Benchmark 评估
│   └── utils/              # LLM API 客户端
├── config/                 # 框架配置
├── data/                   # 数据（TSPLIB 实例、结果）
├── run_pipeline.py         # 框架运行入口
└── test_pipeline.py        # 验证测试
```

## 已收集的 TSP 实现

| 算法 | 类型 | 位置 |
|------|------|------|
| Nearest Neighbor | 经典构造式 | `tsp/classic/` |
| Christofides | 近似算法 | `tsp/approximation/` |
| Branch & Bound | 精确算法 | `tsp/exact/` |
| Held-Karp (DP) | 精确算法 | `tsp/exact/` |
| Concorde (pyconcorde) | 精确算法 | `tsp/exact/pyconcorde/` |
| OR-Tools (CP-SAT) | 精确/混合 | `tsp/or-tools-ref/` |
| 2-opt | 局部搜索 | `tsp/local_search/` |
| LK Heuristic | 局部搜索 | `tsp/local_search/lk_heuristic/` |
| LKH-3 | 局部搜索 | `tsp/local_search/LKH3/` |
| Simulated Annealing | 元启发式 | `tsp/meta/` |
| Ant Colony (ACO) | 元启发式 | `tsp/meta/` |
| Genetic Algorithm (GA) | 元启发式 | `tsp/meta/` |
| PSO | 元启发式 | `tsp/meta/` |
| POMO (RL) | 神经网络 | `tsp/meta/pomo/` |

## CodeEvolution-CO 框架

分析现有求解代码 → 提取算法逻辑 → 构建设计空间 → 演化出新方案。

### 快速开始

```bash
# 1. 收集开源实现
python run_pipeline.py collect

# 2. 用 LLM 分析代码提取算法逻辑
python run_pipeline.py analyze

# 3. 构建设计空间
python run_pipeline.py design-space

# 4. 运行演化
python run_pipeline.py evolve --generations 5

# 5. 全流程
python run_pipeline.py all
```
