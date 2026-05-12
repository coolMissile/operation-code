https://github.com/emrahcimren/cvrptw-optimization.git

https://github.com/chkwon/PyHygese.git #hybrid gs

https://github.com/RomuloOliveira/monte-carlo-cvrp.git 蒙特卡洛

https://github.com/VROOM-Project/vroom-scripts.git问题格式转化

https://github.com/yorak/VeRyPy.git cvrp solver

 01_classic/                 # 经典启发式
│   ├── clarke_wright.py        #经典节约算法
│   ├── monte_carlo_savings.py  #蒙特卡洛节约算法
│   ├── sweep.py                #扫描算法
    ├── rfcs.py                #先定线后分组算法
    |—— lagrangian_3opt.py      #拉格朗日松弛3opt算法
│   └── insertion.py            #插入算法

├── 02_meta/                    # 元启发式
│   ├── aco.py                  # 蚁群算法
    ├── ga-for-cvrp.py          #遗传算法
└── 03_advanced/                # 高级算法
    └── cvrp-lns-solver/        #large neighbor search求解器