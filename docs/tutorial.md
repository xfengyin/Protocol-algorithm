# Protocol-algorithm 使用教程

> **WSN LEACH 协议仿真平台 - 完整教程**

## 目录

1. [项目概述](#1-项目概述)
2. [环境安装](#2-环境安装)
3. [快速入门](#3-快速入门)
4. [核心概念](#4-核心概念)
5. [使用场景](#5-使用场景)
6. [性能调优](#6-性能调优)
7. [常见问题](#7-常见问题)

---

## 1. 项目概述

### 1.1 项目简介

Protocol-algorithm 是一个面向无线传感器网络（WSN）的高性能仿真平台，专注于 LEACH 及其变体协议的仿真、分析与优化。

### 1.2 核心特性

| 特性 | 说明 |
|------|------|
| 多协议支持 | LEACH, LEACH-C, LEACH-EE, LEACH-M |
| AI 增强 | 基于 ML 的智能簇头选择 |
| 高性能 | NumPy 向量化 + KD-Tree 空间索引 |
| 并行仿真 | 多进程参数扫描与批量实验 |
| 可扩展 | SPI 插件架构，自定义协议/能量模型 |
| 可观测 | 全链路指标采集 + 可视化图表 |

### 1.3 项目结构

```
protocol-algorithm/
├── src/                         # 源代码
│   ├── models/                  # 数据模型
│   │   ├── node.py              # 节点模型
│   │   ├── base_station.py      # 基站模型
│   │   ├── cluster_head.py      # 簇头模型
│   │   └── network.py           # 网络核心引擎
│   ├── leach/                   # 协议实现
│   │   ├── base.py              # 协议基类
│   │   ├── classic.py           # 经典 LEACH
│   │   ├── leach_c.py           # LEACH-C
│   │   ├── leach_ee.py          # LEACH-EE
│   │   ├── leach_m.py           # LEACH-M
│   │   └── variants.py          # 协议注册表
│   ├── energy/                  # 能量模型
│   │   ├── models.py            # 多种能量模型
│   │   └── radio_model.py       # 一阶无线电模型
│   ├── ai/                      # AI 簇头选择
│   │   ├── selector.py          # AI 选择器基类
│   │   ├── sklearn_selector.py  # scikit-learn 实现
│   │   ├── pytorch_selector.py  # PyTorch 实现
│   │   ├── feature_engineering.py # 特征工程
│   │   └── trainer.py           # 训练器
│   ├── simulation/              # 仿真引擎
│   │   └── engine.py            # 并行仿真引擎
│   ├── visualization/           # 可视化
│   │   ├── metrics_plots.py     # 指标图表
│   │   ├── animator.py          # 动画
│   │   └── comparison.py        # 对比图表
│   ├── data/                    # 数据处理
│   │   ├── generator.py         # 数据生成
│   │   └── sampler.py           # 采样器
│   └── config/                  # 配置管理
│       └── validator.py         # 配置校验
├── config/                      # 配置文件
│   ├── config.yaml              # 默认配置
│   └── leach_ai.yaml            # AI 配置
├── examples/                    # 示例代码
├── tests/                       # 测试用例
├── docs/                        # 文档
├── cli.py                       # 命令行工具
└── main.py                      # 入口脚本
```

---

## 2. 环境安装

### 2.1 系统要求

- Python 3.9+
- 操作系统：Linux / macOS / Windows

### 2.2 创建虚拟环境

```bash
# 使用 venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows

# 或使用 conda
conda create -n leach python=3.10
conda activate leach
```

### 2.3 安装依赖

```bash
# 安装核心依赖
pip install -r requirements.txt

# 或使用 pyproject.toml
pip install -e .
```

### 2.4 验证安装

```bash
python -c "from src.models.network import Network; print('Installation OK')"
```

### 2.5 依赖说明

| 包 | 用途 |
|----|------|
| numpy | 数值计算与向量化 |
| scipy | KD-Tree 空间索引 |
| matplotlib | 图表绘制 |
| seaborn | 统计可视化 |
| scikit-learn | ML 簇头选择（可选） |
| torch | PyTorch 簇头选择（可选） |
| pyyaml | 配置文件解析 |

---

## 3. 快速入门

### 3.1 最简仿真（5 行代码）

```python
from src.models.network import Network

network = Network(n_nodes=100, area=(0, 100, 0, 100), base_station_pos=(50, 50))
results = network.simulate_network(rounds=1000)

print(f"网络生命周期: {results['network_lifetime']} 轮")
print(f"半网络生命周期: {results['half_network_lifetime']} 轮")
```

### 3.2 完整仿真流程

```python
from src.models.network import Network
from src.energy.radio_model import FirstOrderRadioModel
from src.visualization.metrics_plots import MetricsPlotter

# 1. 创建能量模型
radio_model = FirstOrderRadioModel(
    E_elec=50e-9,
    epsilon_fs=10e-12,
    epsilon_mp=0.0013e-12,
    d_threshold=87.0,
    E_da=5e-9
)

# 2. 创建网络
network = Network(
    n_nodes=100,
    area=(0, 100, 0, 100),
    base_station_pos=(50, 50),
    energy_model=radio_model,
    initial_energy=0.5,
    seed=42
)

# 3. 运行仿真
results = network.simulate_network(
    rounds=5000,
    protocol_name='leach'
)

# 4. 查看结果
print(f"网络生命周期: {results['network_lifetime']} 轮")
print(f"仿真总轮数: {results['total_rounds_simulated']} 轮")
print(f"最终剩余能量: {results['final_energy']:.4f} J")

# 5. 生成图表
plotter = MetricsPlotter()
plotter.plot_all_metrics(results, output_dir='results/plots', show=False)
```

### 3.3 对比不同协议

```python
from src.simulation.engine import BatchSimulator

simulator = BatchSimulator()

comparison = simulator.compare_protocols(
    n_nodes=100,
    area=(0, 100, 0, 100),
    rounds=5000,
    protocols=['leach', 'leach-c', 'leach-ee', 'leach-m'],
    n_runs=5
)

print(f"最优协议: {comparison['summary']['best_protocol']}")
for proto, lifetime in comparison['summary']['lifetimes'].items():
    print(f"  {proto}: {lifetime:.0f} 轮")
```

### 3.4 使用命令行

```bash
# 运行经典 LEACH
python cli.py --config config/config.yaml --output results/leach/

# 运行 AI 增强版本
python cli.py --config config/leach_ai.yaml --output results/ai/

# 指定协议
python cli.py --protocol leach-ee --rounds 5000 --nodes 100
```

---

## 4. 核心概念

### 4.1 网络拓扑

```
         ┌──────────────────────────────────┐
         │         100m × 100m 区域          │
         │                                  │
         │    .  .    . .  .  .    . .      │
         │   .  . CH1 .  .  . CH2 .  .      │
         │  .  /  |  \ . .  / |  \ .  .     │
         │ .  /   |   \ .  /  |   \  .      │
         │ . N1   N2   N3 N4  N5   N6 .     │
         │                                  │
         │            BS (50, 50)           │
         │              *                   │
         └──────────────────────────────────┘
```

- **区域**：默认 100m × 100m 的正方形区域
- **节点**：随机均匀分布在区域内
- **基站**：固定在区域中心 (50, 50)

### 4.2 生命周期阶段

```
Round 0         Round 1         Round 2         ...
┌─────┐        ┌─────┐         ┌─────┐
│Setup│        │Setup│         │Setup│
│Steady│──────►│Steady│───────►│Steady│───────► ...
└─────┘        └─────┘         └─────┘

每轮包含:
  Setup  → 簇头选举 + 簇形成
  Steady → 数据传输 + 能量消耗
```

### 4.3 簇头轮换机制

LEACH 的核心思想是**公平轮换**簇头角色：

```
节点 1:  [普通] [普通] [普通] [CH*]  [普通] [普通] [普通] [CH*]
节点 2:  [CH*]  [普通] [普通] [普通] [普通] [普通] [CH*]  [普通]
节点 3:  [普通] [CH*]  [普通] [普通] [普通] [CH*]  [普通] [普通]
         └──── 1/p = 20 轮 ────┘

每 1/p 轮（默认 20 轮），每个节点恰好成为一次簇头
```

### 4.4 能量消耗模型

```
成员节点:
  发送数据 → E_tx(d, k) = E_elec × k + ε × k × dⁿ

簇头:
  接收数据 → ΣE_rx(k) = Σ(E_elec × k)
  数据聚合 → ΣE_da(k) = Σ(E_DA × k)
  发送到BS → E_tx(d_BS, k_total)
```

### 4.5 配置文件结构

```yaml
# config/config.yaml
network:
  n_nodes: 100          # 节点数量
  area: [0, 100, 0, 100] # 区域范围 (m)
  base_station_pos: [50, 50]
  initial_energy: 0.5   # 初始能量 (J)

energy_model:
  E_elec: 50e-9         # 电路能耗 (J/bit)
  epsilon_fs: 10e-12    # 自由空间系数
  epsilon_mp: 0.0013e-12 # 多径衰落系数
  d_threshold: 87.0     # 距离阈值 (m)
  E_da: 5e-9           # 数据聚合能耗 (J/bit)

simulation:
  rounds: 5000          # 最大轮数
  data_size: 4000       # 数据包大小 (bits)
  seed: 42              # 随机种子

protocol: leach         # 协议选择

leach:
  p: 0.05              # 簇头概率
```

---

## 5. 使用场景

### 5.1 场景 1：基准测试 - 经典 LEACH

```python
from src.models.network import Network

network = Network(n_nodes=100, area=(0, 100, 0, 100), base_station_pos=(50, 50), seed=42)
results = network.simulate_network(rounds=5000, protocol_name='leach')

print(f"网络生命周期: {results['network_lifetime']} 轮")
print(f"半网络生命周期: {results['half_network_lifetime']} 轮")
```

### 5.2 场景 2：能量均衡对比

```python
from src.models.network import Network

# 测试不同协议
protocols = ['leach', 'leach-c', 'leach-ee', 'leach-m']

for proto in protocols:
    network = Network(
        n_nodes=100,
        area=(0, 100, 0, 100),
        base_station_pos=(50, 50),
        seed=42
    )
    results = network.simulate_network(rounds=5000, protocol_name=proto)
    print(f"{proto:>10}: Lifetime={results['network_lifetime']}, "
          f"HalfLife={results['half_network_lifetime']}, "
          f"FinalEnergy={results['final_energy']:.4f}J")
```

### 5.3 场景 3：参数扫描实验

```python
from src.simulation.engine import ParallelSimulationEngine, SimulationConfig

engine = ParallelSimulationEngine(n_workers=4)

base_config = SimulationConfig(
    n_nodes=100,
    area=(0, 100, 0, 100),
    base_station_pos=(50, 50),
    initial_energy=0.5,
    rounds=5000,
    protocol_name='leach',
    seed=42
)

results = engine.run_parameter_sweep(
    base_config=base_config,
    param_grid={
        'n_nodes': [50, 100, 200],
        'initial_energy': [0.3, 0.5, 1.0],
    },
    n_runs=5
)

# 查看聚合结果
for params, data in results['by_params'].items():
    lt = data['network_lifetime']
    print(f"{params}: mean={lt['mean']:.0f}, std={lt['std']:.0f}")
```

### 5.4 场景 4：自定义协议

```python
from src.leach.base import LEACHProtocol
from src.leach.variants import LEACHRegistry
from src.models.network import Network
from src.models.cluster_head import ClusterHead

class MyLEACH(LEACHProtocol):
    def select_cluster_heads(self, network, **kwargs):
        # 自定义逻辑：选择距离基站最近的节点作为簇头
        alive = network.alive_nodes
        bs_pos = network.base_station.position
        n_clusters = max(1, int(len(alive) * self.p))

        # 按到基站距离排序
        sorted_nodes = sorted(alive, key=lambda n: n.distance_to(bs_pos))

        cluster_heads = []
        for i, node in enumerate(sorted_nodes[:n_clusters]):
            node.become_cluster_head(i)
            cluster_heads.append(ClusterHead(node=node, cluster_id=i))

        self.current_round += 1
        return cluster_heads

# 注册协议
LEACHRegistry.register('my_leach', MyLEACH)

# 使用自定义协议
network = Network(n_nodes=100, area=(0, 100, 0, 100), base_station_pos=(50, 50))
results = network.simulate_network(rounds=5000, protocol_name='my_leach')
```

### 5.5 场景 5：自定义能量模型

```python
from src.energy.models import EnergyModel, EnergyModelFactory

class CustomEnergyModel(EnergyModel):
    def __init__(self, E_elec=50e-9, E_da=5e-9):
        self.E_elec = E_elec
        self._E_da = E_da

    @property
    def E_da(self):
        return self._E_da

    def calc_transmit_energy(self, distance, message_size):
        # 自定义能耗公式
        return self.E_elec * message_size * (1 + distance / 100)

    def calc_receive_energy(self, message_size):
        return self.E_elec * message_size

    def get_transmission_mode(self, distance):
        return 'custom'

# 注册模型
EnergyModelFactory.register('custom', CustomEnergyModel)

# 使用自定义模型
radio = EnergyModelFactory.create('custom')
network = Network(
    n_nodes=100,
    area=(0, 100, 0, 100),
    base_station_pos=(50, 50),
    energy_model=radio
)
```

### 5.6 场景 6：AI 增强簇头选择

```python
from src.models.network import Network
from src.data.generator import DataGenerator
from src.ai.trainer import AITrainer
from src.ai.selector import SklearnClusterSelector

# 1. 生成训练数据
generator = DataGenerator(n_nodes=100)
X_train, y_train = generator.generate_balanced_dataset(n_rounds=500)
X_test, y_test = generator.generate_test_dataset(n_rounds=100)

# 2. 训练 AI 模型
trainer = AITrainer(model_type='sklearn', n_estimators=100)
trainer.train(X_train, y_train)

# 3. 评估模型
metrics = trainer.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"F1 Score: {metrics['f1_score']:.3f}")

# 4. 保存模型
trainer.save('models/ai_selector.pkl')

# 5. 使用训练好的模型进行仿真
selector = SklearnClusterSelector(model_path='models/ai_selector.pkl')
selector.load()

# 在仿真中使用 AI 选择器
# 注：需要将 AI 选择器集成到 Network 的 setup_phase 中
```

### 5.7 场景 7：生成可视化图表

```python
from src.models.network import Network
from src.visualization.metrics_plots import MetricsPlotter
from src.visualization.comparison import ProtocolComparator

# 运行仿真
network = Network(n_nodes=100, area=(0, 100, 0, 100), base_station_pos=(50, 50), seed=42)
results = network.simulate_network(rounds=5000, protocol_name='leach')

# 生成所有图表
plotter = MetricsPlotter()
plotter.plot_all_metrics(results, output_dir='results/plots', show=False)

# 生成对比图
comparator = ProtocolComparator()
comparator.compare_protocols(
    {'leach': results_leach, 'leach-c': results_c, 'leach-ee': results_ee},
    output_dir='results/comparison'
)
```

---

## 6. 性能调优

### 6.1 仿真速度优化

| 优化项 | 方法 | 效果 |
|--------|------|------|
| 向量化计算 | 使用 `steady_phase()`（默认） | 3-5x 加速 |
| 空间索引 | KD-Tree 邻居查询 | 2-3x 加速 |
| 随机种子 | 固定 seed 保证可重复 | 调试友好 |
| 并行仿真 | `ParallelSimulationEngine` | N 倍加速（N=CPU 核数） |

```python
# 使用向量化版本（默认）
network.steady_phase()  # 向量化，快速

# 原始版本（用于验证）
network.steady_phase_original()  # 逐节点计算，慢但易调试
```

### 6.2 内存优化

```python
# 减少能量历史记录
class NetworkOptimized(Network):
    def _collect_metrics(self):
        # 只保留关键指标，不存储完整能量历史
        return NetworkMetrics(
            round_number=self.current_round,
            alive_nodes=len(self.alive_nodes),
            dead_nodes=len(self.dead_nodes),
            n_cluster_heads=len(self.cluster_heads),
            total_energy=self.total_energy,
            # 省略 energy_std 等耗时计算
        )
```

### 6.3 参数调优建议

| 参数 | 推荐范围 | 影响 |
|------|----------|------|
| `p`（簇头概率） | 0.03 - 0.10 | 越小簇越大，簇头能耗越高 |
| `n_nodes` | 50 - 500 | 节点越多，仿真越慢 |
| `initial_energy` | 0.3 - 2.0 J | 影响网络生命周期 |
| `data_size` | 2000 - 8000 bits | 影响每轮能耗 |
| `rounds` | 1000 - 10000 | 影响仿真时间 |

### 6.4 协议选择指南

| 场景 | 推荐协议 | 理由 |
|------|----------|------|
| 基准对比 | LEACH | 标准参考 |
| 能量敏感 | LEACH-EE | 能量均衡最优 |
| 集中控制 | LEACH-C | 全局优化 |
| 移动节点 | LEACH-M | 移动性感知 |
| 极致性能 | AI LEACH | 学习最优策略 |

### 6.5 并行配置

```python
# 根据 CPU 核心数自动配置
engine = ParallelSimulationEngine()

# 手动指定
engine = ParallelSimulationEngine(n_workers=4, use_threads=False)

# 使用线程（适合 I/O 密集型）
engine = ParallelSimulationEngine(use_threads=True)
```

---

## 7. 常见问题

### 7.1 安装问题

**Q: `scipy` 安装失败怎么办？**

A: 使用预编译包：
```bash
pip install scipy --only-binary=all
```

**Q: 缺少 `matplotlib` 后端？**

A: 服务器环境使用非交互式后端：
```python
import matplotlib
matplotlib.use('Agg')  # 在 import pyplot 之前设置
```

### 7.2 仿真问题

**Q: 为什么网络生命周期很短？**

A: 可能原因：
1. `initial_energy` 设置过低（建议 ≥ 0.5 J）
2. 基站位置过远（增加 `d_threshold` 或移动基站）
3. `data_size` 过大（减少到 2000-4000 bits）
4. 节点数过多导致簇头频繁死亡

**Q: 簇头数量不稳定？**

A: 检查 `p` 参数。`p=0.05` 时，100 个节点期望簇头数为 5。随机性会导致波动，这是正常的。

**Q: 仿真速度太慢？**

A:
1. 确认使用的是 `steady_phase()` 而非 `steady_phase_original()`
2. 减少 `rounds` 或 `n_nodes`
3. 使用并行仿真引擎
4. 检查是否有大量 print 语句（I/O 是瓶颈）

### 7.3 协议问题

**Q: LEACH-C 和 LEACH 有什么区别？**

A: LEACH 是分布式随机选择，LEACH-C 由基站集中选择能量最高的节点。LEACH-C 通常有更长的网络生命周期，但需要全局信息。

**Q: LEACH-EE 的能量阈值怎么设置？**

A: `energy_threshold` 默认 0.3。设置过高会减少候选簇头数量，设置过低则失去能量感知效果。建议 0.2-0.4 范围内测试。

**Q: 如何添加自定义协议？**

A: 继承 `LEACHProtocol` 基类，实现 `select_cluster_heads` 方法，然后注册：
```python
from src.leach.base import LEACHProtocol
from src.leach.variants import LEACHRegistry

class MyProtocol(LEACHProtocol):
    def select_cluster_heads(self, network, **kwargs):
        # 实现簇头选择逻辑
        ...

LEACHRegistry.register('my_protocol', MyProtocol)
```

### 7.4 AI 问题

**Q: AI 模型训练数据从哪里来？**

A: 使用 `DataGenerator` 通过传统 LEACH 仿真生成标注数据：
```python
from src.data.generator import DataGenerator
gen = DataGenerator(n_nodes=100)
X, y = gen.generate_balanced_dataset(n_rounds=500)
```

**Q: sklearn 和 PyTorch 该选哪个？**

A:
- **sklearn**：快速训练，适合小数据集，推荐初学者
- **PyTorch**：适合大数据集，支持深度学习，需要 GPU 加速

### 7.5 可视化问题

**Q: 图表中文乱码？**

A: 设置中文字体：
```python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
```

**Q: 如何导出高清图片？**

A: 使用高 DPI 设置：
```python
plotter.plot_network_lifetime(results, save_path='plot.png', show=False)
# 默认 DPI=150，可在方法中调整
```

---

## 附录：快速参考卡片

```
╔═══════════════════════════════════════════════════════════╗
║                    快速参考                               ║
╠═══════════════════════════════════════════════════════════╣
║ 创建网络: Network(n_nodes=100, area=(0,100,0,100), ...)  ║
║ 运行仿真: network.simulate_network(rounds=5000)           ║
║ 协议切换: protocol_name='leach-c' / 'leach-ee' / 'leach-m'║
║ 可视化: MetricsPlotter().plot_all_metrics(results, dir)  ║
║ 并行: ParallelSimulationEngine().run_parameter_sweep()   ║
║ 自定义: LEACHRegistry.register('name', ProtocolClass)    ║
╚═══════════════════════════════════════════════════════════╝
```
