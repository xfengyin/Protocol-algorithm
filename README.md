# Protocol-algorithm - LEACH协议仿真与AI优化平台

无线传感器网络（WSN）中 LEACH 协议的仿真与 AI 优化平台。

## 功能特性

- **多 LEACH 变体**：经典 LEACH、LEACH-C、LEACH-EE、LEACH-M
- **真实能量模型**：First Order Radio Model
- **AI 优化**：sklearn / PyTorch 簇头选择器
- **事件驱动仿真**：simpy 风格事件队列
- **并行实验**：多参数组并行 + 自动汇总
- **动态可视化**：matplotlib.animation 分簇动画
- **对比实验**：一键生成多算法对比图
- **配置驱动**：config.yaml 管理所有参数

## 项目架构可视化

### 架构组织图

![项目架构图](https://mdn.alipayobjects.com/one_clip/afts/img/f9-dT6LguPgAAAAAUeAAAAgAoEACAQFr/original)

### 功能思维导图

![功能思维导图](https://mdn.alipayobjects.com/one_clip/afts/img/ymeBQJPJmhgAAAAAXGAAAAgAoEACAQFr/original)

### 仿真流程图

![仿真流程图](https://mdn.alipayobjects.com/one_clip/afts/img/F-t_RpWxOl0AAAAARPAAAAgAoEACAQFr/original)

### 项目质量雷达图

![质量雷达图](https://mdn.alipayobjects.com/one_clip/afts/img/jHpGS48zG70AAAAAUeAAAAgAoEACAQFr/original)

### 技术栈构成

![技术栈饼图](https://mdn.alipayobjects.com/one_clip/afts/img/SKFmSbyXDasAAAAASBAAAAgAoEACAQFr/original)

## 安装

```bash
pip install -e ".[all]"  # 安装全部依赖
pip install -e ".[simulation]"  # 仅仿真依赖
pip install -e ".[ai]"  # AI依赖
```

## 快速开始

```python
from src.models.network import Network
from src.leach.classic import ClassicLEACH
from src.energy.radio_model import FirstOrderRadioModel

# 初始化能量模型
energy_model = FirstOrderRadioModel()

# 创建网络
network = Network(
    n_nodes=100,
    area=(0, 100, 0, 100),
    base_station_pos=(50, 50),
    energy_model=energy_model
)

# 运行仿真
leach = ClassicLEACH(network)
results = network.simulate_network(rounds=5000)

# 查看结果
print(f"网络生命周期: {results['network_lifetime']} 轮")
print(f"总能耗: {results['total_energy']:.2f} J")
```

## 命令行使用

```bash
# 运行经典 LEACH
python main.py --config config/config.yaml --output results/leach/

# 运行 AI 增强 LEACH
python main.py --config config/leach_ai.yaml --output results/leach_ai/

# 对比实验
python examples/compare_variants.py --output results/comparison/
```

## 项目结构

```
Protocol-algorithm/
├── src/
│   ├── leach/          # LEACH 协议变体
│   ├── models/         # 数据模型
│   ├── energy/        # 能量模型
│   ├── ai/            # AI 优化
│   ├── simulation/    # 仿真引擎
│   ├── visualization/ # 可视化
│   └── data/          # 数据生成
├── config/            # 配置文件
├── tests/            # 单元测试
├── docs/             # 文档
└── examples/         # 示例
```

## 算法变体

| 算法 | 描述 |
|------|------|
| LEACH | 经典轮式簇头选择 |
| LEACH-C | 集中式簇头选择（基站优化） |
| LEACH-EE | 能量均衡分簇 |
| LEACH-M | 移动节点支持 |

## 与其他类似项目对比

| 特性 | 本项目 | Slipstream-Max/wsn | Nachi28/WSN_LEACH | Hzz-Git/WSN-Clustering-Benchmark |
|------|--------|--------------------|--------------------|----------------------------------|
| 协议变体 | 4种LEACH变体 | 仅经典LEACH | 仅经典LEACH | 多种协议 |
| AI优化 | sklearn + PyTorch | ❌ | ❌ | 部分支持 |
| 并行仿真 | 多进程+参数扫描 | ❌ | ❌ | ✅ |
| 可视化 | 图表+动画+对比图 | 简单拓扑图 | ❌ | 丰富的图表 |
| 代码质量 | 生产级架构 | 教学演示 | 原型级别 | 科研级 |
| 配置驱动 | YAML配置 | ❌ | ❌ | ✅ |
| 项目规模 | 企业级 | 小型 | 小型 | 中大型 |

### 本项目核心优势

1. **生产级架构**：遵循SOLID原则，分层清晰，支持插件化扩展
2. **多协议+AI**：完整协议仿真与机器学习优化的深度融合
3. **高效仿真**：多进程并行 + 参数扫描，适合批量实验
4. **丰富可视化**：网络拓扑动画、性能指标图、算法对比图

## 可借鉴的开源项目

本项目从以下优秀开源项目中借鉴了设计理念和功能特性：

### 1. WSN-Clustering-Benchmark
**GitHub**: https://github.com/Hzz-Git/WSN-Clustering-Benchmark

**借鉴内容**:
- 多协议对比实验设计
- YAML配置驱动实验参数
- 一键复现论文结果的脚本
- 完整的docs文档结构

**集成方式**: 在 `examples/` 目录添加了 `reproduce.py` 风格的复现脚本

### 2. wsn-cluster-routing-ml
**GitHub**: https://github.com/apturner19/wsn-cluster-routing-ml

**借鉴内容**:
- 使用WSN-DS公开数据集进行模型验证
- 多ML模型对比实验（Neural Network, Random Forest, SVM, Logistic Regression）
- 完善的性能评估指标（Accuracy, Precision, Recall, F1-score）
- 数据预处理流程标准化

**集成方式**: 在 `src/ai/` 目录添加了数据集兼容接口和评估模块

### 3. WSN-Scheduling-with-Reinforcement-Learning
**GitHub**: https://github.com/AliAmini93/WSN-Scheduling-with-Reinforcement-Learning

**借鉴内容**:
- DQN强化学习算法思想
- Stable Baselines3集成
- 智能调度优化策略

**集成方式**: 在 `src/ai/` 目录预留了强化学习接口，便于未来扩展

### 4. 其他项目借鉴

- **Slipstream-Max/wsn**: 简单直观的可视化设计
- **Nachi28/WSN_LEACH**: Excel导出功能集成到结果保存模块
- **LEACH-PY**: 经典协议实现的参考

## 未来扩展方向

基于借鉴内容，计划添加以下功能：

- [ ] 支持WSN-DS公开数据集训练AI模型
- [ ] 添加PDR(包投递率)、E2E延迟等网络性能指标
- [ ] 集成强化学习模块(DQN/PPO)用于智能调度
- [ ] 支持多基站部署场景
- [ ] 添加异构节点能量模型
- [ ] 完善交互式Jupyter Notebook演示
- [ ] 添加更多对比基准协议(HEED, PEGASIS等)

## 可视化结果示例

### 网络生命周期曲线

运行仿真后自动生成存活节点数随时间变化的曲线图：

```python
from src.visualization.metrics_plots import MetricsPlotter

plotter = MetricsPlotter()
plotter.plot_network_lifetime(results)
```

### 能量消耗分析

生成的能量消耗分布图，包含总能耗和每轮能耗变化率。

### 簇分布统计

每轮簇头数量分布和簇大小分布的统计图表。

### 动态动画

使用 `Animator` 类生成网络拓扑变化的动画：

```python
from src.visualization.animator import NetworkAnimator

animator = NetworkAnimator(network)
animator.animate(num_frames=100, interval=100)
```

## License

MIT License
