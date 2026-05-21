# API 参考文档

> **WSN LEACH Protocol Simulation - Core API Reference**

## 目录

1. [核心类总览](#1-核心类总览)
2. [网络模型 (models)](#2-网络模型-models)
3. [LEACH 协议 (leach)](#3-leach-协议-leach)
4. [能量模型 (energy)](#4-能量模型-energy)
5. [AI 簇头选择 (ai)](#5-ai-簇头选择-ai)
6. [仿真引擎 (simulation)](#6-仿真引擎-simulation)
7. [数据处理 (data)](#7-数据处理-data)
8. [可视化 (visualization)](#8-可视化-visualization)
9. [配置管理 (config)](#9-配置管理-config)
10. [扩展接口](#10-扩展接口)

---

## 1. 核心类总览

| 模块 | 类名 | 用途 | 文件 |
|------|------|------|------|
| `models` | `Node` | 传感器节点 | `src/models/node.py` |
| `models` | `BaseStation` | 基站 | `src/models/base_station.py` |
| `models` | `ClusterHead` | 簇头 | `src/models/cluster_head.py` |
| `models` | `Network` | 网络仿真引擎 | `src/models/network.py` |
| `models` | `NetworkMetrics` | 网络指标 | `src/models/network.py` |
| `models` | `NodeRole` | 节点角色枚举 | `src/models/node.py` |
| `leach` | `LEACHProtocol` | 协议抽象基类 | `src/leach/base.py` |
| `leach` | `ClassicLEACH` | 经典 LEACH | `src/leach/classic.py` |
| `leach` | `LEACHC` | LEACH-C 集中式 | `src/leach/leach_c.py` |
| `leach` | `LEACHEE` | LEACH-EE 能量均衡 | `src/leach/leach_ee.py` |
| `leach` | `LEACHM` | LEACH-M 移动节点 | `src/leach/leach_m.py` |
| `leach` | `LEACHRegistry` | 协议注册表 | `src/leach/variants.py` |
| `energy` | `EnergyModel` | 能量模型抽象基类 | `src/energy/models.py` |
| `energy` | `FirstOrderRadioModel` | 一阶无线电模型 | `src/energy/models.py` |
| `energy` | `Mica2Model` | Mica2 硬件模型 | `src/energy/models.py` |
| `energy` | `RssiBasedModel` | RSSI 模型 | `src/energy/models.py` |
| `energy` | `AdaptiveEnergyModel` | 自适应模型 | `src/energy/models.py` |
| `energy` | `EnergyModelFactory` | 能量模型工厂 | `src/energy/models.py` |
| `ai` | `AIClusterSelector` | AI 选择器基类 | `src/ai/selector.py` |
| `ai` | `EnsembleClusterSelector` | 集成选择器 | `src/ai/selector.py` |
| `ai` | `AITrainer` | 训练管道 | `src/ai/trainer.py` |
| `ai` | `AdvancedFeatureExtractor` | 高级特征提取器 | `src/ai/feature_engineering.py` |
| `simulation` | `ParallelSimulationEngine` | 并行仿真引擎 | `src/simulation/engine.py` |
| `simulation` | `BatchSimulator` | 批量仿真器 | `src/simulation/engine.py` |
| `data` | `DataGenerator` | 训练数据生成器 | `src/data/generator.py` |
| `visualization` | `MetricsPlotter` | 指标图表绘制器 | `src/visualization/metrics_plots.py` |
| `visualization` | `ComparisonPlotter` | 对比图表绘制器 | `src/visualization/comparison.py` |

---

## 2. 网络模型 (models)

### 2.1 Node

传感器节点数据类。

```python
from src.models.node import Node, NodeRole
```

#### 构造函数

```python
Node(
    id: int,                          # 节点唯一标识
    x: float,                         # X 坐标 (m)
    y: float,                         # Y 坐标 (m)
    initial_energy: float = 0.5,      # 初始能量 (J)
    energy: float = 0.5,              # 当前剩余能量 (J)
    role: NodeRole = NodeRole.NORMAL  # 角色
)
```

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `id` | `int` | 节点唯一标识 |
| `x` | `float` | X 坐标 (m) |
| `y` | `float` | Y 坐标 (m) |
| `initial_energy` | `float` | 初始能量 (J) |
| `energy` | `float` | 当前剩余能量 (J) |
| `role` | `NodeRole` | 角色：NORMAL / CLUSTER_HEAD / DEAD |
| `cluster_id` | `Optional[int]` | 所属簇 ID |
| `cluster_head` | `Optional[Node]` | 所属簇头节点 |
| `energy_history` | `List[float]` | 能量历史记录 |
| `transmissions` | `int` | 总发送次数 |
| `receptions` | `int` | 总接收次数 |

#### 只读属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `is_alive` | `bool` | 是否存活（energy > 0 且 role != DEAD） |
| `is_cluster_head` | `bool` | 是否为簇头 |
| `is_normal` | `bool` | 是否为普通节点 |

#### 方法

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `distance_to(other)` | `other: Node \| Tuple[float, float]` | `float` | 计算到另一个节点或坐标的距离 |
| `consume_energy(amount)` | `amount: float` | `None` | 消耗指定能量，能量 ≤ 0 时自动标记死亡 |
| `become_cluster_head(cluster_id)` | `cluster_id: int` | `None` | 成为簇头 |
| `join_cluster(cluster_head, cluster_id)` | `cluster_head: Node, cluster_id: int` | `None` | 加入簇 |
| `leave_cluster()` | - | `None` | 离开簇 |
| `reset_role()` | - | `None` | 重置角色为普通节点 |
| `get_features(base_station_pos)` | `base_station_pos: Tuple[float, float]` | `np.ndarray` | 获取 AI 特征向量 (7 维) |

#### 示例

```python
node = Node(id=0, x=25.0, y=30.0, initial_energy=0.5)

# 消耗能量
node.consume_energy(0.1)
print(node.energy)  # 0.4

# 成为簇头
node.become_cluster_head(0)
print(node.is_cluster_head)  # True

# 计算距离
other = Node(id=1, x=50.0, y=50.0)
dist = node.distance_to(other)
print(f"Distance: {dist:.2f} m")

# 获取特征
features = node.get_features((50.0, 50.0))
print(features)  # [25.0, 30.0, 0.4, 0.8, 25.0, 0, 0]
```

---

### 2.2 BaseStation

基站模型。

```python
from src.models.base_station import BaseStation
```

#### 构造函数

```python
BaseStation(
    x: float,    # X 坐标
    y: float     # Y 坐标
)
```

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `x` | `float` | X 坐标 |
| `y` | `float` | Y 坐标 |
| `position` | `Tuple[float, float]` | 位置元组 |
| `total_received_data` | `int` | 累计接收数据量 (bits) |
| `cluster_heads_history` | `List[List[int]]` | 每轮簇头 ID 记录 |

#### 方法

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `distance_to(node)` | `node: Node \| Tuple[float, float]` | `float` | 计算到节点的距离 |
| `receive_data(size_bits)` | `size_bits: int` | `None` | 记录接收数据量 |
| `record_round(cluster_head_ids)` | `cluster_head_ids: List[int]` | `None` | 记录当前轮簇头 ID |
| `reset()` | - | `None` | 重置基站状态 |

---

### 2.3 ClusterHead

簇头数据类。

```python
from src.models.cluster_head import ClusterHead
```

#### 构造函数

```python
ClusterHead(
    node: Node,                 # 对应的节点
    cluster_id: int,            # 簇 ID
    member_nodes: List[Node]    # 成员节点列表
)
```

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `node` | `Node` | 对应的节点对象 |
| `cluster_id` | `int` | 簇 ID |
| `member_nodes` | `List[Node]` | 成员节点列表 |
| `n_members` | `int` | 成员数量（只读） |
| `total_member_energy` | `float` | 成员总能量（只读） |
| `average_member_energy` | `float` | 成员平均能量（只读） |

#### 方法

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `add_member(node)` | `node: Node` | `None` | 添加成员节点 |
| `remove_member(node)` | `node: Node` | `None` | 移除成员节点 |
| `clear_members()` | - | `None` | 清空所有成员 |

---

### 2.4 Network

网络仿真核心引擎。

```python
from src.models.network import Network, NetworkMetrics
```

#### 构造函数

```python
Network(
    n_nodes: int,                                    # 节点数量
    area: Tuple[float, float, float, float],        # 区域 (x_min, x_max, y_min, y_max)
    base_station_pos: Tuple[float, float],          # 基站位置
    energy_model: Optional[EnergyModel] = None,     # 能量模型（默认 FirstOrderRadioModel）
    initial_energy: float = 0.5,                    # 初始能量 (J)
    seed: Optional[int] = None                      # 随机种子
)
```

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `n_nodes` | `int` | 总节点数 |
| `area` | `Tuple[float, float, float, float]` | 区域范围 |
| `initial_energy` | `float` | 初始能量 |
| `energy_model` | `EnergyModel` | 使用的能量模型 |
| `base_station` | `BaseStation` | 基站对象 |
| `nodes` | `List[Node]` | 所有节点列表 |
| `cluster_heads` | `List[ClusterHead]` | 当前簇头列表 |
| `current_round` | `int` | 当前轮数 |
| `metrics_history` | `List[NetworkMetrics]` | 历史指标 |

#### 只读属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `alive_nodes` | `List[Node]` | 存活节点列表 |
| `dead_nodes` | `List[Node]` | 死亡节点列表 |
| `n_alive` | `int` | 存活节点数 |
| `total_energy` | `float` | 网络总能量 |

#### 方法

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `get_neighbors(node, threshold_distance)` | `node: Node, threshold_distance: float` | `List[Node]` | 使用 KD-Tree 查询邻居节点 |
| `get_neighbors_vectorized(threshold_distance)` | `threshold_distance: float` | `Dict[int, List[int]]` | 批量获取所有节点的邻居 |
| `reset_nodes()` | - | `None` | 重置所有节点角色 |
| `setup_phase(protocol_name, **kwargs)` | `protocol_name: str` | `List[ClusterHead]` | 执行设置阶段（簇头选举+簇形成） |
| `steady_phase(data_size)` | `data_size: int = 4000` | `None` | 执行稳定阶段（数据传输，向量化版本） |
| `steady_phase_original(data_size)` | `data_size: int = 4000` | `None` | 原始实现版本（用于对比） |
| `simulate_round(protocol_name, **kwargs)` | `protocol_name: str` | `NetworkMetrics` | 模拟一轮 |
| `simulate_network(rounds, protocol_name, stop_condition, **kwargs)` | `rounds: int, protocol_name: str` | `Dict[str, Any]` | 模拟整个网络生命周期 |
| `reset()` | - | `None` | 重置整个网络 |
| `get_node(node_id)` | `node_id: int` | `Optional[Node]` | 根据 ID 获取节点 |
| `_collect_metrics()` | - | `NetworkMetrics` | 收集当前网络指标 |

#### simulate_network 返回值

```python
{
    "rounds": List[int],                 # 每轮编号
    "alive_nodes": List[int],            # 每轮存活节点数
    "dead_nodes": List[int],             # 每轮死亡节点数
    "total_energy": List[float],         # 每轮总能量
    "cluster_heads": List[int],          # 每轮簇头数
    "network_lifetime": int,             # 第一个节点死亡轮数
    "half_network_lifetime": int,        # 一半节点死亡轮数
    "total_rounds_simulated": int,       # 实际仿真轮数
    "final_energy": float,               # 最终剩余能量
    "first_dead_round": int,             # 首个节点死亡轮数
    "half_dead_round": int,              # 半数节点死亡轮数
}
```

#### 示例

```python
from src.models.network import Network

network = Network(
    n_nodes=100,
    area=(0, 100, 0, 100),
    base_station_pos=(50, 50),
    initial_energy=0.5,
    seed=42
)

results = network.simulate_network(rounds=5000, protocol_name='leach')

print(f"生命周期: {results['network_lifetime']} 轮")
print(f"半生命周期: {results['half_network_lifetime']} 轮")
print(f"最终能量: {results['final_energy']:.4f} J")
```

---

### 2.5 NetworkMetrics

每轮网络指标数据类。

```python
from src.models.network import NetworkMetrics
```

#### 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `round_number` | `int` | 当前轮数 |
| `alive_nodes` | `int` | 存活节点数 |
| `dead_nodes` | `int` | 死亡节点数 |
| `n_cluster_heads` | `int` | 簇头数量 |
| `total_energy` | `float` | 总能量 |
| `average_energy` | `float` | 平均能量 |
| `energy_std` | `float` | 能量标准差 |
| `cluster_size_distribution` | `Dict[int, int]` | 簇大小分布 |

#### 方法

| 方法 | 返回值 | 说明 |
|------|--------|------|
| `to_dict()` | `Dict[str, Any]` | 转换为字典用于序列化 |

---

### 2.6 NodeRole

节点角色枚举。

```python
from src.models.node import NodeRole

NodeRole.NORMAL        # "normal" - 普通节点
NodeRole.CLUSTER_HEAD  # "cluster_head" - 簇头
NodeRole.DEAD          # "dead" - 死亡
```

---

## 3. LEACH 协议 (leach)

### 3.1 LEACHProtocol（抽象基类）

所有 LEACH 协议的基类。

```python
from src.leach.base import LEACHProtocol
```

#### 构造函数

```python
LEACHProtocol(p: float = 0.05)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `p` | `float` | `0.05` | 簇头选择概率 |

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `p` | `float` | 簇头概率 |
| `current_round` | `int` | 当前轮数 |

#### 方法

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `select_cluster_heads(network, **kwargs)` | `network: Network` | `List[ClusterHead]` | **[抽象方法]** 选择簇头 |
| `get_threshold(n_iterations)` | `n_iterations: int` | `float` | 计算阈值 T(n) |
| `reset()` | - | `None` | 重置协议状态 |

#### get_threshold 实现

```python
def get_threshold(self, n_iterations: int) -> float:
    if n_iterations % int(1 / self.p) == 0:
        return self.p
    return self.p * (1 - self.p * (n_iterations % int(1 / self.p)))
```

---

### 3.2 ClassicLEACH

经典 LEACH 协议实现。

```python
from src.leach.classic import ClassicLEACH
```

#### 构造函数

```python
ClassicLEACH(p: float = 0.05)
```

#### 核心逻辑

1. 计算目标簇头数 `n_clusters = n_alive * p`
2. 打乱存活节点顺序
3. 对每个节点计算阈值 `T(n)`
4. 随机数 < T(n) 的节点成为簇头
5. 达到目标簇头数后停止

#### 示例

```python
from src.leach.classic import ClassicLEACH

protocol = ClassicLEACH(p=0.05)
cluster_heads = protocol.select_cluster_heads(network)
```

---

### 3.3 LEACHC

集中式 LEACH-C 协议。

```python
from src.leach.leach_c import LEACHC
```

#### 构造函数

```python
LEACHC(p: float = 0.05)
```

#### 核心逻辑

1. 收集所有节点能量和位置
2. 计算能量比率 `score = energy / initial_energy`
3. 按能量分数降序排序
4. 选择前 `n_clusters` 个节点作为簇头

#### 示例

```python
from src.leach.leach_c import LEACHC

protocol = LEACHC(p=0.05)
cluster_heads = protocol.select_cluster_heads(network)
```

---

### 3.4 LEACHEE

能量均衡 LEACH-EE 协议。

```python
from src.leach.leach_ee import LEACHEE
```

#### 构造函数

```python
LEACHEE(
    p: float = 0.05,              # 基础簇头概率
    energy_threshold: float = 0.3, # 能量阈值
    density_weight: float = 0.3    # 密度权重
)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `p` | `float` | `0.05` | 基础簇头概率 |
| `energy_threshold` | `float` | `0.3` | 低于此值不参与选举 |
| `density_weight` | `float` | `0.3` | 密度因子权重 |

#### 核心逻辑

```
adjusted_T = base_T × energy_factor × density_factor

energy_factor = energy / initial_energy
density_factor = 1 - (density / max_density) × density_weight
```

---

### 3.5 LEACHM

移动节点 LEACH-M 协议。

```python
from src.leach.leach_m import LEACHM
```

#### 构造函数

```python
LEACHM(
    p: float = 0.05,                       # 簇头概率
    mobility_speed_range: tuple = (0, 1.0) # 移动速度范围 (m/s)
)
```

#### 核心逻辑

```
mobility_penalty = 1 - (speed / max_speed) × 0.5
adjusted_T = base_T × mobility_penalty
```

#### 额外方法

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `update_positions(network)` | `network: Network` | `None` | 更新移动节点位置 |
| `predict_next_position(node)` | `node: Node` | `Tuple[float, float]` | 预测下一位置 |

---

### 3.6 LEACHRegistry

协议注册表，支持 SPI 式扩展。

```python
from src.leach.variants import LEACHRegistry
```

#### 类方法

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `register(name, protocol_class)` | `name: str, protocol_class: Type[LEACHProtocol]` | `None` | 注册新协议 |
| `get(name)` | `name: str` | `LEACHProtocol` | 获取协议实例 |
| `list_protocols()` | - | `List[str]` | 列出所有已注册协议 |

#### 内置协议

| 名称 | 类 | 说明 |
|------|-----|------|
| `leach` | `ClassicLEACH` | 经典 LEACH |
| `classic` | `ClassicLEACH` | 经典 LEACH（别名） |
| `leach-c` | `LEACHC` | 集中式 LEACH-C |
| `leach_c` | `LEACHC` | 集中式 LEACH-C（别名） |
| `leach-ee` | `LEACHEE` | 能量均衡 LEACH-EE |
| `leach_ee` | `LEACHEE` | 能量均衡 LEACH-EE（别名） |
| `leach-m` | `LEACHM` | 移动 LEACH-M |
| `leach_m` | `LEACHM` | 移动 LEACH-M（别名） |

#### 示例

```python
# 列出所有协议
print(LEACHRegistry.list_protocols())

# 获取协议实例
protocol = LEACHRegistry.get('leach-c')

# 注册自定义协议
class MyProtocol(LEACHProtocol):
    def select_cluster_heads(self, network, **kwargs):
        # 自定义逻辑
        ...

LEACHRegistry.register('my_protocol', MyProtocol)
```

---

## 4. 能量模型 (energy)

### 4.1 EnergyModel（抽象基类）

能量模型接口。

```python
from src.energy.models import EnergyModel
```

#### 抽象方法

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `calc_transmit_energy(distance, message_size)` | `distance: float, message_size: int` | `float` | 计算发送能耗 (J) |
| `calc_receive_energy(message_size)` | `message_size: int` | `float` | 计算接收能耗 (J) |
| `get_transmission_mode(distance)` | `distance: float` | `str` | 获取传输模式名称 |

#### 抽象属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `E_da` | `float` | 数据聚合能耗系数 (J/bit) |

---

### 4.2 FirstOrderRadioModel

经典一阶无线电能量模型。

```python
from src.energy.models import FirstOrderRadioModel
# 或
from src.energy.radio_model import FirstOrderRadioModel
```

#### 构造函数

```python
FirstOrderRadioModel(
    E_elec: float = 50e-9,          # 电路能耗 (J/bit)
    epsilon_fs: float = 10e-12,     # 自由空间系数 (J/bit/m²)
    epsilon_mp: float = 0.0013e-12, # 多径衰落系数 (J/bit/m⁴)
    d_threshold: float = 87.0,      # 距离阈值 (m)
    E_da: float = 5e-9              # 数据聚合能耗 (J/bit)
)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `E_elec` | `float` | `50e-9` | 发射/接收电路能耗 |
| `epsilon_fs` | `float` | `10e-12` | 自由空间放大器系数 |
| `epsilon_mp` | `float` | `0.0013e-12` | 多径衰落放大器系数 |
| `d_threshold` | `float` | `87.0` | 自由空间 ↔ 多径切换距离 |
| `E_da` | `float` | `5e-9` | 数据聚合能耗系数 |

#### 方法

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `calc_transmit_energy(distance, message_size)` | `distance: float, message_size: int` | `float` | 计算发送能耗 |
| `calc_receive_energy(message_size)` | `message_size: int` | `float` | 计算接收能耗 |
| `calc_aggregation_energy(message_size)` | `message_size: int` | `float` | 计算聚合能耗 |
| `calc_total_communication_energy(tx_distance, rx_distance, message_size)` | 三个参数 | `float` | 计算总通信能耗 |
| `get_transmission_mode(distance)` | `distance: float` | `str` | 返回 `'free_space'` 或 `'multi_path'` |
| `estimate_network_energy_budget(n_nodes, n_rounds, avg_cluster_size, message_size)` | 四个参数 | `float` | 估算网络总能耗预算 |
| `to_dict()` | - | `dict` | 转换为字典 |
| `from_dict(data)` | `data: dict` | `FirstOrderRadioModel` | 从字典创建实例 |

#### 示例

```python
radio = FirstOrderRadioModel()

# 计算 30m 距离发送 4000 bits 的能耗
energy = radio.calc_transmit_energy(30.0, 4000)
print(f"TX Energy: {energy:.2e} J")

# 计算接收 4000 bits 的能耗
energy = radio.calc_receive_energy(4000)
print(f"RX Energy: {energy:.2e} J")

# 判断传输模式
mode = radio.get_transmission_mode(100.0)
print(f"Mode: {mode}")  # multi_path

# 序列化/反序列化
data = radio.to_dict()
radio2 = FirstOrderRadioModel.from_dict(data)
```

---

### 4.3 Mica2Model

Mica2 硬件节点能量模型。

```python
from src.energy.models import Mica2Model
```

#### 构造函数

```python
Mica2Model(
    Eelec_tx: float = 60e-9,     # TX 电路能耗
    Eelec_rx: float = 45e-9,     # RX 电路能耗
    Efs: float = 10e-12,         # 自由空间系数
    Emp: float = 0.0013e-12,     # 多径衰落系数
    d0: float = 87.0,            # 距离阈值
    E_da: float = 5e-9           # 数据聚合能耗
)
```

**特点**：TX 和 RX 电路能耗分离（60nJ vs 45nJ），更贴近 Mica2 硬件实测。

---

### 4.4 RssiBasedModel

基于 RSSI 的能量模型。

```python
from src.energy.models import RssiBasedModel
```

#### 构造函数

```python
RssiBasedModel(
    E_elec: float = 50e-9,           # 电路能耗
    E_da: float = 5e-9,              # 数据聚合能耗
    reference_rssi: float = -30.0,   # 参考 RSSI (dBm)
    path_loss_exponent: float = 2.0, # 路径损耗指数
    noise_floor: float = -100.0      # 噪声底限 (dBm)
)
```

#### 额外方法

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `estimate_distance_from_rssi(rssi)` | `rssi: float` | `float` | 从 RSSI 估算距离 |

---

### 4.5 AdaptiveEnergyModel

自适应能量模型，根据网络状态动态调参。

```python
from src.energy.models import AdaptiveEnergyModel
```

#### 构造函数

```python
AdaptiveEnergyModel(
    base_model: Optional[EnergyModel] = None,  # 基础模型
    energy_factor: float = 1.0,                # 能量因子
    adapt_threshold: float = 0.2               # 自适应阈值
)
```

#### 方法

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `update_adapt_factor(avg_energy_ratio)` | `avg_energy_ratio: float` | `None` | 根据平均能量比率更新自适应因子 |
| `reset()` | - | `None` | 重置累积能耗 |

**自适应逻辑**：
- `avg_energy_ratio < 0.2` → `factor = 0.8`（节能模式）
- `avg_energy_ratio > 0.8` → `factor = 1.2`（高性能模式）
- 其他 → `factor = 1.0`（标准模式）

---

### 4.6 EnergyModelFactory

能量模型工厂，支持注册和创建。

```python
from src.energy.models import EnergyModelFactory
```

#### 类方法

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `register(name, model_class)` | `name: str, model_class: Type[EnergyModel]` | `None` | 注册新模型 |
| `create(model_name, **kwargs)` | `model_name: str` | `EnergyModel` | 创建模型实例 |
| `list_models()` | - | `List[str]` | 列出所有已注册模型 |

#### 内置模型

| 名称 | 类 |
|------|-----|
| `first_order` | `FirstOrderRadioModel` |
| `mica2` | `Mica2Model` |
| `rssi` | `RssiBasedModel` |
| `adaptive` | `AdaptiveEnergyModel` |

#### 示例

```python
# 列出所有模型
print(EnergyModelFactory.list_models())

# 创建模型
radio = EnergyModelFactory.create('first_order', E_elec=50e-9)

# 注册自定义模型
EnergyModelFactory.register('custom', CustomEnergyModel)
radio = EnergyModelFactory.create('custom')
```

---

## 5. AI 簇头选择 (ai)

### 5.1 AIClusterSelector（抽象基类）

AI 簇头选择器基类。

```python
from src.ai.selector import AIClusterSelector
```

#### 构造函数

```python
AIClusterSelector(model_path: Optional[str] = None)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `model_path` | `Optional[str]` | 模型保存路径 |

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `model` | `Any` | 底层 ML 模型 |
| `model_path` | `Optional[str]` | 模型路径 |
| `is_trained` | `bool` | 是否已训练 |
| `feature_names` | `List[str]` | 特征名称列表 |

#### 抽象方法

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `predict(features)` | `features: np.ndarray` | `np.ndarray` | 预测簇头分数 |
| `train(X, y, **kwargs)` | `X: np.ndarray, y: np.ndarray` | `None` | 训练模型 |
| `_save_model(path)` | `path: Path` | `None` | 保存模型实现 |
| `_load_model(path)` | `path: Path` | `None` | 加载模型实现 |

#### 实例方法

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `select_cluster_heads(network, n_clusters, **kwargs)` | `network: Network` | `List[ClusterHead]` | 基于 AI 模型选择簇头 |
| `_extract_features(network)` | `network: Network` | `np.ndarray` | 提取节点特征 |
| `save(path)` | `path: Optional[str]` | `None` | 保存模型 |
| `load(path)` | `path: Optional[str]` | `None` | 加载模型 |
| `explain_prediction(node, network)` | `node: Node, network: Network` | `Dict[str, Any]` | 解释预测结果 |
| `evaluate(X, y_true, threshold)` | `X: np.ndarray, y_true: np.ndarray, threshold: float = 0.5` | `Dict[str, float]` | 评估模型性能 |

#### evaluate 返回值

```python
{
    "accuracy": float,       # 总体准确率
    "precision": float,      # 精确度
    "recall": float,         # 召回率
    "f1_score": float,       # F1 分数
    "true_positives": int,   # 真正例
    "true_negatives": int,   # 真反例
    "false_positives": int,  # 假正例
    "false_negatives": int,  # 假反例
}
```

---

### 5.2 EnsembleClusterSelector

集成簇头选择器，支持多模型加权投票。

```python
from src.ai.selector import EnsembleClusterSelector
```

#### 构造函数

```python
EnsembleClusterSelector(selectors: Optional[List[AIClusterSelector]] = None)
```

#### 方法

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `add_selector(selector, weight)` | `selector: AIClusterSelector, weight: float` | `None` | 添加选择器 |
| `set_weights(weights)` | `weights: List[float]` | `None` | 设置权重 |
| `predict_scores(features)` | `features: np.ndarray` | `np.ndarray` | 获取加权平均分数 |
| `select_cluster_heads(network, n_clusters, **kwargs)` | `network: Network` | `List[ClusterHead]` | 集成选择簇头 |
| `get_selector_info()` | - | `List[Dict[str, Any]]` | 获取选择器信息 |

#### 加权公式

```
score = Σ(w_i × score_i) / Σ(w_i)
```

#### 示例

```python
ensemble = EnsembleClusterSelector()
ensemble.add_selector(sklearn_selector, weight=0.6)
ensemble.add_selector(pytorch_selector, weight=0.4)

cluster_heads = ensemble.select_cluster_heads(network, n_clusters=5)

# 查看选择器信息
info = ensemble.get_selector_info()
# [{'type': 'SklearnClusterSelector', 'weight': 0.6, 'trained': True}, ...]
```

---

### 5.3 AITrainer

AI 训练管道。

```python
from src.ai.trainer import AITrainer
```

#### 构造函数

```python
AITrainer(model_type: str = "sklearn", **kwargs)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_type` | `str` | `"sklearn"` | 模型类型：`sklearn` 或 `pytorch` |
| `**kwargs` | - | - | 传递给选择器的参数 |

#### 方法

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `load_dataset(path)` | `path: str` | `Tuple[np.ndarray, np.ndarray]` | 从 CSV 加载数据集 |
| `prepare_data(X, y, test_size, random_state)` | 四个参数 | `Tuple[X_train, X_test, y_train, y_test]` | 划分训练/测试集（分层采样） |
| `train(X, y, epochs, **kwargs)` | 参数见签名 | `Dict[str, Any]` | 训练模型 |
| `evaluate(X, y)` | `X: np.ndarray, y: np.ndarray` | `Dict[str, float]` | 评估模型 |
| `save(path)` | `path: str` | `None` | 保存模型 |
| `load(path)` | `path: str` | `None` | 加载模型 |
| `get_feature_importance()` | - | `Dict[str, float]` | 获取特征重要性 |

#### train 返回值

```python
{
    "accuracy": float,      # 训练准确率
    "n_samples": int,       # 样本数
    "positive_ratio": float # 正例比例
}
```

#### evaluate 返回值

```python
{
    "accuracy": float,   # 准确率
    "precision": float,  # 精确度
    "recall": float,     # 召回率
    "f1": float          # F1 分数
}
```

#### 示例

```python
from src.ai.trainer import AITrainer
from src.data.generator import DataGenerator

# 生成数据
gen = DataGenerator(n_nodes=100)
X, y = gen.generate_balanced_dataset(n_rounds=500)

# 训练
trainer = AITrainer(model_type='sklearn', n_estimators=100)
result = trainer.train(X, y)
print(f"训练准确率: {result['accuracy']:.3f}")

# 评估
metrics = trainer.evaluate(X, y)
print(f"F1: {metrics['f1']:.3f}")

# 保存/加载
trainer.save('models/ai_selector.pkl')
trainer.load('models/ai_selector.pkl')
```

---

### 5.4 AdvancedFeatureExtractor

高级特征提取器（19 维特征）。

```python
from src.ai.feature_engineering import AdvancedFeatureExtractor
```

#### 构造函数

```python
AdvancedFeatureExtractor(
    network: Network,   # 网络对象
    n_bins: int = 10    # 直方图分箱数
)
```

#### 方法

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `compute_stats()` | - | `FeatureStats` | 计算全局统计信息 |
| `get_stats()` | - | `FeatureStats` | 获取统计信息（惰性计算） |
| `extract_features(node)` | `node: Node` | `np.ndarray` | 提取单个节点的 19 维特征 |
| `extract_batch(nodes)` | `nodes: List[Node]` | `np.ndarray` | 批量提取特征 |
| `get_feature_names()` | - | `List[str]` | 获取特征名称列表 |
| `get_importance_weights()` | - | `Dict[str, float]` | 获取特征重要性权重 |
| `apply_weights(features, weights)` | `features: np.ndarray, weights: Optional[Dict]` | `np.ndarray` | 应用特征权重 |
| `normalize_features(features, method)` | `features: np.ndarray, method: str` | `np.ndarray` | 归一化特征 |
| `clear_cache()` | - | `None` | 清除缓存 |

#### 特征名称列表（19 维）

```python
[
    'x', 'y', 'dist_to_center',
    'energy', 'energy_ratio', 'energy_zscore', 'energy_rank',
    'dist_to_bs', 'dist_to_bs_normalized',
    'neighbor_count_20', 'neighbor_count_40', 'neighbor_count_60',
    'avg_neighbor_energy', 'max_neighbor_energy',
    'total_transmissions', 'total_receptions', 'comm_load',
    'is_near_bs', 'is_in_center'
]
```

#### 归一化方法

| 方法 | 公式 | 说明 |
|------|------|------|
| `zscore` | `(x - μ) / σ` | 标准 Z-Score 归一化 |
| `minmax` | `(x - min) / (max - min)` | Min-Max 归一化 |
| `robust` | `(x - median) / IQR` | 抗异常值归一化 |

---

### 5.5 FeatureSelector

特征选择器。

```python
from src.ai.feature_engineering import FeatureSelector
```

#### 构造函数

```python
FeatureSelector(extractor: AdvancedFeatureExtractor)
```

#### 方法

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `select_by_variance(features, threshold)` | `features: np.ndarray, threshold: float` | `Tuple[np.ndarray, List[int]]` | 基于方差筛选特征 |
| `select_by_correlation(features, threshold)` | `features: np.ndarray, threshold: float` | `Tuple[np.ndarray, List[int]]` | 基于相关性筛选（去除高相关特征） |
| `select_top_k(features, k, importance_scores)` | `features: np.ndarray, k: int` | `Tuple[np.ndarray, List[int]]` | 选择 Top-K 重要特征 |

---

### 5.6 FeatureStats

特征统计数据类。

```python
from src.ai.feature_engineering import FeatureStats
```

#### 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `mean_energy` | `float` | 平均能量 |
| `std_energy` | `float` | 能量标准差 |
| `center_x` | `float` | 网络中心 X |
| `center_y` | `float` | 网络中心 Y |
| `n_alive` | `int` | 存活节点数 |
| `density` | `float` | 节点密度 |
| `energy_histogram` | `np.ndarray` | 能量直方图 |

---

## 6. 仿真引擎 (simulation)

### 6.1 ParallelSimulationEngine

并行仿真引擎。

```python
from src.simulation.engine import ParallelSimulationEngine, SimulationConfig, SimulationResult
```

#### SimulationConfig

仿真配置数据类。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_nodes` | `int` | `100` | 节点数 |
| `area` | `Tuple[float, float, float, float]` | `(0, 100, 0, 100)` | 区域 |
| `base_station_pos` | `Tuple[float, float]` | `(50, 50)` | 基站位置 |
| `initial_energy` | `float` | `0.5` | 初始能量 (J) |
| `rounds` | `int` | `1000` | 最大轮数 |
| `protocol_name` | `str` | `"leach"` | 协议名 |
| `seed` | `Optional[int]` | `None` | 随机种子 |

#### SimulationResult

仿真结果数据类。

| 字段 | 类型 | 说明 |
|------|------|------|
| `config` | `SimulationConfig` | 使用的配置 |
| `run_id` | `int` | 运行 ID |
| `results` | `Dict[str, Any]` | 仿真结果字典 |
| `execution_time` | `float` | 执行时间 (s) |

#### 构造函数

```python
ParallelSimulationEngine(
    n_workers: Optional[int] = None,   # 工作进程数（默认 CPU 核数-1）
    use_threads: bool = False          # 是否使用线程而非进程
)
```

#### 方法

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `run_parameter_sweep(base_config, param_grid, n_runs, aggregate)` | 见签名 | `Dict[str, Any]` | 运行参数扫描实验 |
| `run_parallel_experiments(configs, show_progress)` | `configs: List[SimulationConfig]` | `List[SimulationResult]` | 并行运行多个配置 |
| `_aggregate_results(results, param_names, param_values)` | 三个参数 | `Dict[str, Any]` | 聚合结果（内部方法） |

#### run_parameter_sweep 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `base_config` | `SimulationConfig` | 基础配置 |
| `param_grid` | `Dict[str, List[Any]]` | 参数网格，如 `{'n_nodes': [50, 100, 200]}` |
| `n_runs` | `int` | 每个参数组合的运行次数 |
| `aggregate` | `bool` | 是否聚合结果 |

#### 示例

```python
engine = ParallelSimulationEngine(n_workers=4)

config = SimulationConfig(
    n_nodes=100,
    area=(0, 100, 0, 100),
    base_station_pos=(50, 50),
    rounds=5000,
    protocol_name='leach',
    seed=42
)

results = engine.run_parameter_sweep(
    base_config=config,
    param_grid={
        'n_nodes': [50, 100, 200],
        'initial_energy': [0.3, 0.5, 1.0],
    },
    n_runs=5
)
```

---

### 6.2 BatchSimulator

批量仿真器。

```python
from src.simulation.engine import BatchSimulator
```

#### 构造函数

```python
BatchSimulator(engine: Optional[ParallelSimulationEngine] = None)
```

#### 方法

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `compare_protocols(n_nodes, area, rounds, protocols, n_runs)` | 见签名 | `Dict[str, Any]` | 对比不同协议 |
| `save_results(results, output_path)` | `results: Dict, output_path: str` | `None` | 保存结果到 JSON |
| `load_results(input_path)` | `input_path: str` | `Dict[str, Any]` | 从 JSON 加载结果 |

#### compare_protocols 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_nodes` | `int` | `100` | 节点数 |
| `area` | `Tuple[float, float, float, float]` | `(0, 100, 0, 100)` | 区域 |
| `rounds` | `int` | `1000` | 轮数 |
| `protocols` | `Optional[List[str]]` | `['leach', 'leach_c', 'leach_ee', 'leach_m']` | 协议列表 |
| `n_runs` | `int` | `10` | 每个协议运行次数 |

#### 返回值

```python
{
    'protocols': {
        'leach': {...},    # 每个协议的仿真结果
        'leach_c': {...},
        ...
    },
    'summary': {
        'best_protocol': str,       # 最优协议名
        'lifetimes': {              # 各协议生命周期
            'leach': float,
            ...
        }
    }
}
```

---

## 7. 数据处理 (data)

### 7.1 DataGenerator

训练数据生成器。

```python
from src.data.generator import DataGenerator
```

#### 构造函数

```python
DataGenerator(
    n_nodes: int = 100,                        # 节点数
    area: Tuple[float, float, float, float] = (0, 100, 0, 100),  # 区域
    base_station_pos: Tuple[float, float] = (50, 50),            # 基站位置
    initial_energy: float = 0.5                # 初始能量
)
```

#### 方法

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `generate_training_data(n_rounds, protocol_name, seed)` | 见签名 | `pd.DataFrame` | 生成训练数据 DataFrame |
| `generate_balanced_dataset(n_rounds, positive_ratio, seed)` | 见签名 | `Tuple[np.ndarray, np.ndarray]` | 生成平衡数据集 (X, y) |
| `save_dataset(X, y, path, feature_names)` | 见签名 | `None` | 保存数据集为 CSV |

#### generate_training_data 返回 DataFrame 列

| 列名 | 类型 | 说明 |
|------|------|------|
| `round` | `int` | 轮次 |
| `node_id` | `int` | 节点 ID |
| `x` | `float` | X 坐标 |
| `y` | `float` | Y 坐标 |
| `energy` | `float` | 当前能量 |
| `energy_ratio` | `float` | 能量比例 |
| `dist_to_bs` | `float` | 到基站距离 |
| `is_alive` | `bool` | 是否存活 |
| `is_cluster_head` | `bool` | 是否为簇头 |
| `round_dead` | `Optional[int]` | 死亡轮次 |
| `n_neighbors` | `int` | 邻居数量 |

#### generate_balanced_dataset 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_rounds` | `int` | `500` | 模拟轮数 |
| `positive_ratio` | `float` | `0.1` | 正例（簇头）比例 |
| `seed` | `Optional[int]` | `None` | 随机种子 |

#### 示例

```python
gen = DataGenerator(n_nodes=100)

# 生成原始数据
df = gen.generate_training_data(n_rounds=500, seed=42)
print(df.head())

# 生成平衡数据集
X, y = gen.generate_balanced_dataset(n_rounds=500, positive_ratio=0.1)
print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"正例比例: {y.mean():.3f}")

# 保存
gen.save_dataset(X, y, 'data/training.csv',
                 feature_names=['x', 'y', 'energy', 'energy_ratio', 'dist_to_bs', 'n_neighbors'])
```

---

## 8. 可视化 (visualization)

### 8.1 MetricsPlotter

指标图表绘制器。

```python
from src.visualization.metrics_plots import MetricsPlotter
```

#### 构造函数

```python
MetricsPlotter(style: str = "seaborn-v0_8-darkgrid")
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `style` | `str` | `"seaborn-v0_8-darkgrid"` | matplotlib 绘图风格 |

#### 方法

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `plot_network_lifetime(results, save_path, show)` | 见签名 | `None` | 绘制网络生命周期图 |
| `plot_energy_consumption(results, save_path, show)` | 见签名 | `None` | 绘制能量消耗图（双面板） |
| `plot_cluster_distribution(results, save_path, show)` | 见签名 | `None` | 绘制簇头分布图 |
| `plot_all_metrics(results, output_dir, show)` | 见签名 | `None` | 绘制所有指标并保存 |

#### plot_network_lifetime 说明

- 绘制存活节点数随轮次变化的曲线
- 标注半死亡线（红色虚线）和 10% 存活线（橙色虚线）
- 标注第一个节点死亡轮次（灰色点线 + 注释）

#### plot_energy_consumption 说明

- 左面板：总能量随轮次变化
- 右面板：每轮能量消耗柱状图

#### 示例

```python
plotter = MetricsPlotter()

# 单独绘制
plotter.plot_network_lifetime(results, save_path='results/lifetime.png', show=False)
plotter.plot_energy_consumption(results, save_path='results/energy.png', show=False)
plotter.plot_cluster_distribution(results, save_path='results/clusters.png', show=False)

# 一键绘制所有
plotter.plot_all_metrics(results, output_dir='results/plots', show=False)
# 生成: network_lifetime.png, energy_consumption.png, cluster_distribution.png
```

---

### 8.2 ComparisonPlotter

协议对比图表绘制器。

```python
from src.visualization.comparison import ComparisonPlotter
```

#### 构造函数

```python
ComparisonPlotter()
```

#### 方法

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `compare_protocols(results_list, protocol_names, save_path, show)` | 见签名 | `None` | 绘制多协议对比图（2×2 面板） |
| `_plot_performance_comparison(results_list, protocol_names, ax)` | 三个参数 | `None` | 绘制性能指标对比柱状图 |
| `generate_comparison_report(results_list, protocol_names, output_dir)` | 三个参数 | `None` | 生成对比报告（图表 + Markdown） |

#### compare_protocols 面板

| 面板 | 内容 |
|------|------|
| 左上 | 网络生命周期对比（存活节点数曲线） |
| 右上 | 总能耗对比 |
| 左下 | 性能指标对比柱状图（FND, HND, Total Rounds） |
| 右下 | 节点存活率曲线 (%) |

#### 示例

```python
comparator = ComparisonPlotter()

# 绘制对比图
comparator.compare_protocols(
    results_list=[results_leach, results_c, results_ee],
    protocol_names=['LEACH', 'LEACH-C', 'LEACH-EE'],
    save_path='results/comparison.png',
    show=False
)

# 生成完整报告
comparator.generate_comparison_report(
    results_list=[results_leach, results_c, results_ee],
    protocol_names=['LEACH', 'LEACH-C', 'LEACH-EE'],
    output_dir='results/comparison'
)
# 生成: comparison.png + comparison_report.md
```

---

## 9. 配置管理 (config)

### 9.1 ConfigValidator

配置校验器。

```python
from src.config.validator import ConfigValidator
```

#### 功能

- 验证 YAML 配置文件的格式和值范围
- 确保能量参数、网络参数、协议参数的合理性
- 提供默认值回退

#### 示例配置

```yaml
# config/config.yaml
network:
  n_nodes: 100
  area: [0, 100, 0, 100]
  base_station_pos: [50, 50]
  initial_energy: 0.5

energy_model:
  E_elec: 50e-9
  epsilon_fs: 10e-12
  epsilon_mp: 0.0013e-12
  d_threshold: 87.0
  E_da: 5e-9

simulation:
  rounds: 5000
  data_size: 4000
  seed: 42

protocol: leach

leach:
  p: 0.05

output:
  directory: ./results
  save_plots: true
  save_animation: false
```

---

## 10. 扩展接口

### 10.1 自定义协议

继承 `LEACHProtocol` 并注册：

```python
from src.leach.base import LEACHProtocol
from src.leach.variants import LEACHRegistry
from src.models.cluster_head import ClusterHead

class MyProtocol(LEACHProtocol):
    def select_cluster_heads(self, network, **kwargs):
        # 实现簇头选择逻辑
        alive = network.alive_nodes
        n_clusters = max(1, int(len(alive) * self.p))
        
        # 自定义选择逻辑...
        cluster_heads = []
        # ...
        
        self.current_round += 1
        return cluster_heads

# 注册
LEACHRegistry.register('my_protocol', MyProtocol)
```

### 10.2 自定义能量模型

实现 `EnergyModel` 接口并注册：

```python
from src.energy.models import EnergyModel, EnergyModelFactory

class CustomModel(EnergyModel):
    @property
    def E_da(self) -> float:
        return 5e-9

    def calc_transmit_energy(self, distance, message_size):
        # 自定义公式
        ...

    def calc_receive_energy(self, message_size):
        ...

    def get_transmission_mode(self, distance):
        ...

# 注册
EnergyModelFactory.register('custom', CustomModel)
```

### 10.3 自定义 AI 选择器

继承 `AIClusterSelector`：

```python
from src.ai.selector import AIClusterSelector

class CustomSelector(AIClusterSelector):
    def predict(self, features):
        # 返回分数数组
        ...

    def train(self, X, y, **kwargs):
        # 训练逻辑
        ...

    def _save_model(self, path):
        # 保存实现
        ...

    def _load_model(self, path):
        # 加载实现
        ...
```

---

## 附录：类型速查

### 常用类型别名

```python
from typing import List, Dict, Tuple, Optional, Any

# 坐标类型
Position = Tuple[float, float]

# 区域类型
Area = Tuple[float, float, float, float]  # (x_min, x_max, y_min, y_max)

# 特征矩阵
FeatureMatrix = np.ndarray  # (n_samples, n_features)

# 标签数组
LabelArray = np.ndarray     # (n_samples,)

# 仿真结果
SimulationResult = Dict[str, Any]
```

### 常量参考

| 常量 | 值 | 说明 |
|------|-----|------|
| `E_elec` | `50e-9` J/bit | 电路能耗 |
| `ε_fs` | `10e-12` J/bit/m² | 自由空间系数 |
| `ε_mp` | `0.0013e-12` J/bit/m⁴ | 多径衰落系数 |
| `d₀` | `87.0` m | 距离阈值 |
| `E_DA` | `5e-9` J/bit | 数据聚合能耗 |
| `p` | `0.05` | 默认簇头概率 |
| `initial_energy` | `0.5` J | 默认初始能量 |
| `data_size` | `4000` bits | 默认数据包大小 |
