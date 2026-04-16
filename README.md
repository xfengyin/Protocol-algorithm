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

## License

MIT License
