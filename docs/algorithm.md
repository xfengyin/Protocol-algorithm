# LEACH 协议算法原理详解

> **Wireless Sensor Network (WSN) LEACH Protocol Simulation**

## 目录

1. [概述](#1-概述)
2. [网络模型](#2-网络模型)
3. [能量模型](#3-能量模型)
4. [经典 LEACH 协议](#4-经典-leach-协议)
5. [LEACH 变体](#5-leach-变体)
6. [AI 簇头选择器](#6-ai-簇头选择器)
7. [仿真引擎](#7-仿真引擎)
8. [性能指标](#8-性能指标)
9. [伪代码](#9-伪代码)
10. [参考资料](#10-参考资料)

---

## 1. 概述

LEACH（Low-Energy Adaptive Clustering Hierarchy，低功耗自适应聚类层次）是一种专为无线传感器网络设计的分层路由协议。它由 MIT 的 Wendi Heinzelman 等人于 2000 年提出，主要目标是：

- **均衡能量消耗**：通过随机轮换簇头（Cluster Head），使每个节点均匀分担能耗
- **延长网络生命周期**：避免个别节点因过度通信而过早死亡
- **数据聚合**：簇头对成员数据进行压缩聚合，减少向基站发送的数据量

### 1.1 网络架构

```
                    基站 (Base Station)
                         |
                    ┌────┴────┐
               簇头 CH1    簇头 CH2
               /  |  \      /  |  \
           节点  节点 节点  节点  节点 节点
```

- **普通节点**：仅向所属簇头发送数据
- **簇头 (CH)**：接收簇内数据，进行数据聚合后发送至基站
- **基站 (BS)**：汇聚全网数据的最终接收点

### 1.2 轮结构 (Round)

LEACH 以"轮"为单位运行，每轮分为两个阶段：

| 阶段 | 名称 | 主要操作 | 时长占比 |
|------|------|----------|----------|
| 1 | 设置阶段 (Setup Phase) | 簇头选举、簇形成、TDMA 调度 | ~10% |
| 2 | 稳定阶段 (Steady Phase) | 数据传输、数据聚合、多路复用 | ~90% |

---

## 2. 网络模型

### 2.1 节点模型 (Node)

每个传感器节点具备以下属性和状态：

```python
@dataclass
class Node:
    id: int                          # 节点唯一标识
    x: float                         # X 坐标 (m)
    y: float                         # Y 坐标 (m)
    initial_energy: float = 0.5      # 初始能量 (J)
    energy: float = 0.5              # 当前剩余能量 (J)
    role: NodeRole                   # 角色: NORMAL / CLUSTER_HEAD / DEAD
    cluster_id: Optional[int]        # 所属簇 ID
    energy_history: List[float]      # 能量历史记录
    transmissions: int               # 发送次数
    receptions: int                  # 接收次数
```

#### 2.1.1 节点状态机

```
┌──────────┐   消耗能量至 0   ┌──────┐
│  NORMAL  │ ──────────────► │ DEAD │
│ (普通节点) │                │(死亡) │
└────┬─────┘                └──────┘
     │ 被选为簇头
     ▼
┌──────────────┐
│ CLUSTER_HEAD │
│  (簇头)      │
└──────────────┘
```

### 2.2 基站模型 (BaseStation)

基站位于网络的固定位置，负责接收所有簇头聚合后的数据：

```python
class BaseStation:
    x: float                         # X 坐标
    y: float                         # Y 坐标
    position: Tuple[float, float]    # 位置元组
    total_received_data: int         # 累计接收数据量 (bits)
    cluster_heads_history: List      # 每轮簇头记录
```

### 2.3 簇头模型 (ClusterHead)

簇头是每轮选举产生的临时角色，负责管理簇内成员：

```python
@dataclass
class ClusterHead:
    node: Node                       # 对应的节点
    cluster_id: int                  # 簇 ID
    member_nodes: List[Node]         # 成员节点列表
```

---

## 3. 能量模型

### 3.1 First Order Radio Model（一阶无线电模型）

LEACH 使用经典的 First Order Radio Model 来计算通信过程中的能量消耗。该模型将能耗分为**电路能耗**和**放大器能耗**两部分。

#### 3.1.1 发送能耗

发送 k bits 数据到距离 d 处的能耗：

```
                    E_elec × k + ε_fs × k × d²,    当 d ≤ d₀（自由空间模型）
E_tx(k, d) = {
                    E_elec × k + ε_mp × k × d⁴,    当 d > d₀（多径衰落模型）
```

#### 3.1.2 接收能耗

接收 k bits 数据的能耗：

```
E_rx(k) = E_elec × k
```

#### 3.1.3 数据聚合能耗

簇头对接收到的数据进行融合处理的能耗：

```
E_da(k) = E_DA × k
```

#### 3.1.4 关键参数表

| 参数 | 含义 | 默认值 | 单位 |
|------|------|--------|------|
| E_elec | 发射/接收电路能耗 | 50 × 10⁻⁹ | J/bit |
| ε_fs | 自由空间放大器系数 | 10 × 10⁻¹² | J/bit/m² |
| ε_mp | 多径衰落放大器系数 | 0.0013 × 10⁻¹² | J/bit/m⁴ |
| d₀ (d_threshold) | 距离阈值（自由空间 ↔ 多径切换点） | 87.0 | m |
| E_DA | 数据聚合能耗系数 | 5 × 10⁻⁹ | J/bit |

#### 3.1.5 距离阈值推导

距离阈值 d₀ 是自由空间模型 (d²) 与多径衰落模型 (d⁴) 的交叉点：

```
d₀ = √(ε_fs / ε_mp) = √(10×10⁻¹² / 0.0013×10⁻¹²) ≈ 87.7 m
```

#### 3.1.6 能量计算示例

**场景 1：节点发送 4000 bits 数据到 30m 外的簇头**

```
d = 30m ≤ d₀ = 87m → 使用自由空间模型

E_tx = (50×10⁻⁹ + 10×10⁻¹² × 30²) × 4000
     = (50×10⁻⁹ + 9×10⁻⁹) × 4000
     = 59×10⁻⁹ × 4000
     = 2.36 × 10⁻⁴ J
```

**场景 2：簇头发送 20000 bits 聚合数据到 100m 外的基站**

```
d = 100m > d₀ = 87m → 使用多径衰落模型

E_tx = (50×10⁻⁹ + 0.0013×10⁻¹² × 100⁴) × 20000
     = (50×10⁻⁹ + 1.3×10⁻⁸) × 20000
     = 6.3×10⁻⁸ × 20000
     = 1.26 × 10⁻³ J
```

### 3.2 支持的扩展能量模型

系统采用面向接口设计，支持多种能量模型通过 `EnergyModelFactory` 注册：

| 模型 | 说明 | 特点 |
|------|------|------|
| `FirstOrderRadioModel` | 经典一阶无线电模型 | 标准 LEACH 使用 |
| `Mica2Model` | Mica2 硬件节点模型 | TX/RX 电路能耗分离 |
| `RssiBasedModel` | 基于 RSSI 的模型 | 适用于实测场景 |
| `AdaptiveEnergyModel` | 自适应模型 | 根据网络状态动态调参 |

---

## 4. 经典 LEACH 协议

### 4.1 簇头选择算法

LEACH 使用**分布式随机选择**算法来决定哪些节点在当轮成为簇头。

#### 4.1.1 阈值函数 T(n)

每个节点 n 在轮次 r 生成随机数 r ∈ [0, 1)，若 r < T(n) 则成为簇头：

```
         p / (1 - p × (r mod (1/p))),  如果 n ∈ G
T(n) = {
         0,                              否则
```

其中：
- **p**：期望的簇头比例（默认 0.05，即每 100 个节点期望选出 5 个簇头）
- **r**：当前轮次编号
- **G**：在过去 1/p 轮中**未**成为簇头的节点集合

#### 4.1.2 公式解析

```
第一轮 (r=0):    T(n) = p / (1 - p×0) = p = 0.05
第二轮 (r=1):    T(n) = p / (1 - p×1) = p / (1-p) ≈ 0.0526
第三轮 (r=2):    T(n) = p / (1 - p×2) = p / (1-2p) ≈ 0.0556
...
第 1/p 轮:       所有未当选节点必定成为簇头
```

**关键特性**：该公式保证了每 1/p 轮（即 20 轮）内，**每个节点恰好成为一次簇头**，从而实现能量消耗的公平分配。

#### 4.1.3 实现逻辑

```python
def get_threshold(self, n_iterations: int) -> float:
    if n_iterations % int(1 / self.p) == 0:
        return self.p
    return self.p * (1 - self.p * (n_iterations % int(1 / self.p)))
```

### 4.2 簇形成过程

1. 簇头选择完成后，每个簇头广播 `ADVERTISE-CLUSTER-HEAD` 消息
2. 非簇头节点根据接收信号强度选择**最近的簇头**
3. 节点向所选簇头发送 `JOIN-CLUSTER` 消息
4. 簇头创建 TDMA 调度表并广播给成员

### 4.3 稳定阶段数据传输

```
成员节点 ──发送数据──► 簇头 ──接收并聚合──► 簇头 ──发送聚合数据──► 基站
   │                      │                      │
   │ E_tx(d, k)           │ ΣE_rx(k)             │ E_tx(d, k_total)
   │                      │ + ΣE_da(k)           │
   ▼                      ▼                      ▼
消耗能量              消耗能量               消耗能量
```

#### 簇头的总能耗

```
E_CH = (N/k - 1) × E_rx(k) + (N/k) × E_da(k) + E_tx(k_total, d_BS)
```

其中：
- N：总节点数
- k：簇头数量
- d_BS：簇头到基站的距离

#### 普通节点的能耗

```
E_nonCH = E_tx(k, d_CH)
```

其中 d_CH 为节点到所属簇头的距离。

---

## 5. LEACH 变体

系统内置 4 种 LEACH 变体，可通过协议注册机制扩展：

### 5.1 经典 LEACH (Classic LEACH)

- **文件**：`src/leach/classic.py`
- **策略**：纯分布式随机选择
- **优点**：简单、低开销
- **缺点**：可能出现簇头分布不均、高能耗节点被选中

### 5.2 LEACH-C (Centralized LEACH)

- **文件**：`src/leach/leach_c.py`
- **策略**：由基站集中优化簇头选择
- **算法**：
  1. 收集所有节点的能量和位置信息
  2. 计算每个节点的能量比率：`score = node.energy / node.initial_energy`
  3. 按能量分数降序排列
  4. 选择能量最高的 N×p 个节点作为簇头
- **优势**：
  - 全局最优选择
  - 能量感知，避免低能量节点成为簇头
  - 簇分布更均匀
- **代码核心**：
  ```python
  node_info.sort(key=lambda x: x['score'], reverse=True)
  ```

### 5.3 LEACH-EE (Energy-Efficient LEACH)

- **文件**：`src/leach/leach_ee.py`
- **策略**：能量均衡 + 邻居密度感知
- **动态阈值调整公式**：
  ```
  T_adjusted = T_base × energy_factor × density_factor
  ```

  其中：
  - `energy_factor = node.energy / node.initial_energy`（剩余能量比例）
  - `density_factor = 1 - (density / max_density) × density_weight`
  - 密度阈值：30m
  - 密度权重：0.3

- **能量门槛**：能量比率 < 0.3 的节点不参与选举
- **优势**：
  - 高能量节点更易成为簇头
  - 稀疏区域节点更易成为簇头（避免密集区过多簇头）
  - 能量消耗更均衡

### 5.4 LEACH-M (Mobile LEACH)

- **文件**：`src/leach/leach_m.py`
- **策略**：支持移动节点的 LEACH
- **移动性处理**：
  1. 每轮更新节点位置（随机方向 + 随机速度）
  2. 记录历史位置用于预测
  3. 对高速移动节点施加选择惩罚
- **移动惩罚公式**：
  ```
  mobility_penalty = 1 - (speed / max_speed) × 0.5
  T_adjusted = T_base × mobility_penalty
  ```
- **特性**：
  - 移动速度越快，成为簇头的概率越低
  - 支持位置预测（基于历史位置）
  - 自动裁剪边界（clip to area）

### 5.5 协议注册表 (LEACHRegistry)

所有协议通过注册表管理，支持 SPI 式插件扩展：

```python
LEACHRegistry.register('my_protocol', MyLEACHProtocol)
```

| 协议名 | 类名 | 适用场景 |
|--------|------|----------|
| `leach` / `classic` | `ClassicLEACH` | 基准对比 |
| `leach-c` / `leach_c` | `LEACHC` | 集中优化 |
| `leach-ee` / `leach_ee` | `LEACHEE` | 能量均衡 |
| `leach-m` / `leach_m` | `LEACHM` | 移动场景 |

---

## 6. AI 簇头选择器

### 6.1 架构设计

AI 簇头选择器采用抽象基类设计，支持多种 ML 后端：

```
AIClusterSelector (ABC)
├── SklearnClusterSelector    # scikit-learn 后端
├── PyTorchClusterSelector    # PyTorch 后端
└── EnsembleClusterSelector   # 集成选择器
```

### 6.2 特征工程

系统提供 19 维高级特征向量：

| 类别 | 特征名 | 说明 | 权重 |
|------|--------|------|------|
| **位置** | `x`, `y` | 节点坐标 | - |
| | `dist_to_center` | 到网络中心的距离 | - |
| **能量** | `energy` | 当前能量 | - |
| | `energy_ratio` | 剩余能量比例 | **2.5** |
| | `energy_zscore` | 能量 Z-Score | **2.0** |
| | `energy_rank` | 能量排名归一化 | **2.0** |
| **距离** | `dist_to_bs` | 到基站距离 | - |
| | `dist_to_bs_normalized` | 归一化到基站距离 | **1.5** |
| **邻居** | `neighbor_count_20/40/60` | 不同半径邻居数 | - |
| | `avg_neighbor_energy` | 邻居平均能量 | **1.0** |
| | `max_neighbor_energy` | 邻居最大能量 | - |
| **通信** | `total_transmissions` | 总发送次数 | - |
| | `total_receptions` | 总接收次数 | - |
| | `comm_load` | 通信负载 | **0.5** |
| **区域** | `is_near_bs` | 是否靠近基站 (<30m) | **0.8** |
| | `is_in_center` | 是否在网络中心 (<25m) | - |

### 6.3 特征归一化

支持三种归一化方法：

```python
# Z-Score 标准化
x' = (x - μ) / σ

# Min-Max 归一化
x' = (x - min) / (max - min)

# Robust 归一化（抗异常值）
x' = (x - median) / IQR
```

### 6.4 选择流程

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌────────────┐
│ 提取特征     │────►│ 模型推理      │────►│ 分数排序      │────►│ 选择 Top-K │
│ 19维向量     │     │ predict()    │     │ argsort(-1)  │     │ 作为簇头    │
└─────────────┘     └──────────────┘     └──────────────┘     └────────────┘
```

### 6.5 集成选择器

支持多模型加权投票：

```python
ensemble = EnsembleClusterSelector()
ensemble.add_selector(sklearn_selector, weight=0.6)
ensemble.add_selector(pytorch_selector, weight=0.4)

# 加权分数
score = Σ(w_i × score_i) / Σ(w_i)
```

### 6.6 模型评估

提供标准分类指标：

| 指标 | 公式 | 含义 |
|------|------|------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | 总体准确率 |
| Precision | TP/(TP+FP) | 簇头选择精确度 |
| Recall | TP/(TP+FN) | 簇头召回率 |
| F1-Score | 2×P×R/(P+R) | 调和平均 |

---

## 7. 仿真引擎

### 7.1 Network 类

核心仿真引擎 `Network` 类负责管理整个 WSN 生命周期：

```python
network = Network(
    n_nodes=100,                    # 节点数量
    area=(0, 100, 0, 100),         # 区域范围 (m)
    base_station_pos=(50, 50),     # 基站位置
    energy_model=radio_model,       # 能量模型
    initial_energy=0.5,            # 初始能量 (J)
    seed=42                         # 随机种子
)

results = network.simulate_network(
    rounds=5000,                    # 最大轮数
    protocol_name='leach',         # 协议名称
)
```

### 7.2 向量化优化

系统在稳态阶段使用 NumPy 向量化计算：

```python
# 所有节点到所有簇头的距离矩阵 (N × K)
distances = np.linalg.norm(
    alive_positions[:, np.newaxis] - ch_positions[np.newaxis, :],
    axis=2
)

# 最近簇头索引
nearest_ch_indices = np.argmin(distances, axis=1)
```

### 7.3 空间索引

使用 KD-Tree 加速邻居查询：

```python
from scipy.spatial import cKDTree

# 构建空间索引
self._spatial_index = cKDTree(positions)

# O(log N) 范围查询
neighbors = index.query_ball_point((x, y), radius)
```

### 7.4 并行仿真

支持多进程参数扫描：

```python
engine = ParallelSimulationEngine(n_workers=4)
results = engine.run_parameter_sweep(
    base_config=config,
    param_grid={'n_nodes': [50, 100, 200]},
    n_runs=5
)
```

---

## 8. 性能指标

### 8.1 核心指标

| 指标 | 定义 | 计算公式 | 优劣 |
|------|------|----------|------|
| **网络生命周期** | 第一个节点死亡时的轮数 | FND (First Node Dies) | ↑ 越好 |
| **半网络生命周期** | 50% 节点死亡时的轮数 | HND (Half Node Dies) | ↑ 越好 |
| **总能耗** | 网络累计能量消耗 | Σ(E_initial - E_current) | ↓ 越好 |
| **能耗均衡度** | 节点能量的标准差 | σ(E_i) | ↓ 越好 |
| **簇头分布** | 每轮簇头数量 | count(CH per round) | → 稳定越好 |

### 8.2 NetworkMetrics 数据类

```python
@dataclass
class NetworkMetrics:
    round_number: int
    alive_nodes: int
    dead_nodes: int
    n_cluster_heads: int
    total_energy: float
    average_energy: float
    energy_std: float
    cluster_size_distribution: Dict[int, int]
```

### 8.3 可视化

```python
plotter = MetricsPlotter()
plotter.plot_network_lifetime(results, save_path='lifetime.png')
plotter.plot_energy_consumption(results, save_path='energy.png')
plotter.plot_cluster_distribution(results, save_path='clusters.png')
```

---

## 9. 伪代码

### 9.1 经典 LEACH 主流程

```
ALGORITHM: Classic LEACH
INPUT: N nodes, rounds R, probability p
OUTPUT: simulation results

INITIALIZE network with N nodes
FOR round r = 0 TO R-1 DO
    ── Setup Phase ──
    FOR each alive node n DO
        IF n has not been CH in last 1/p rounds THEN
            threshold T(n) = p / (1 - p × (r mod 1/p))
            random_value = uniform(0, 1)
            IF random_value < T(n) THEN
                n.become_cluster_head()
            END IF
        END IF
    END FOR
    
    ── Steady Phase ──
    FOR each non-CH node n DO
        find nearest cluster_head CH
        n.join_cluster(CH)
        energy = E_tx(distance(n, CH), data_size)
        n.consume_energy(energy)
    END FOR
    
    FOR each cluster_head CH DO
        total_bits = data_size × CH.n_members
        rx_energy = E_rx(total_bits)
        CH.consume_energy(rx_energy)
        
        fusion_energy = E_DA × total_bits
        CH.consume_energy(fusion_energy)
        
        tx_energy = E_tx(distance(CH, BS), total_bits)
        CH.consume_energy(tx_energy)
    END FOR
    
    COLLECT metrics
END FOR

RETURN results with network_lifetime, half_lifetime
```

### 9.2 LEACH-C 集中式选择

```
ALGORITHM: LEACH-C
INPUT: network, probability p
OUTPUT: cluster_heads list

alive_nodes = network.alive_nodes
n_clusters = len(alive_nodes) × p

FOR each node n in alive_nodes DO
    score = n.energy / n.initial_energy
END FOR

SORT nodes by score DESCENDING

FOR i = 0 TO n_clusters-1 DO
    selected_nodes[i].become_cluster_head(i)
END FOR

RETURN cluster_heads
```

### 9.3 LEACH-EE 能量均衡

```
ALGORITHM: LEACH-EE
INPUT: network, probability p, energy_threshold, density_weight
OUTPUT: cluster_heads list

FOR each node n in alive_nodes DO
    neighbors = network.get_neighbors(n, radius=30)
    density[n.id] = len(neighbors)
END FOR

max_density = max(density.values())

FOR each node n in alive_nodes DO
    energy_ratio = n.energy / n.initial_energy
    
    IF energy_ratio < energy_threshold THEN
        CONTINUE  // 跳过低能量节点
    END IF
    
    base_T = get_threshold(current_round)
    energy_factor = energy_ratio
    density_factor = 1 - (density[n.id] / max_density) × density_weight
    
    adjusted_T = base_T × energy_factor × density_factor
    
    IF random() < adjusted_T THEN
        n.become_cluster_head()
    END IF
END FOR

RETURN cluster_heads
```

### 9.4 完整仿真流程

```
FLOWCHART: LEACH Simulation Flow

┌───────────────┐
│  初始化网络    │
│  N 节点       │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  选择协议     │
│  (leach/C/EE/M)│
└───────┬───────┘
        │
        ▼
┌───────────────┐     ┌───────────────┐
│  设置阶段      │────►│  簇头选择     │
│  Setup Phase  │     │  (算法变体)   │
└───────┬───────┘     └───────────────┘
        │
        ▼
┌───────────────┐
│  簇形成       │
│  成员加入     │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  稳定阶段      │
│  Steady Phase │
└───────┬───────┘
        │
        ├─ 成员 → 簇头 (发送数据)
        ├─ 簇头 聚合数据
        └─ 簇头 → 基站 (发送聚合数据)
        │
        ▼
┌───────────────┐
│  收集指标     │
│  Metrics      │
└───────┬───────┘
        │
        ▼
┌───────────────┐     YES    ┌───────────────┐
│  检查停止条件 │───────────►│  输出结果     │
│  所有节点死亡? │            │  分析/可视化  │
└───────┬───────┘            └───────────────┘
        │ NO
        ▼
┌───────────────┐
│  下一轮       │
│  r = r + 1   │
└───────────────┘
```

---

## 10. 参考资料

1. Heinzelman, W. B., Chandrakasan, A. P., & Balakrishnan, H. "An Application-Specific Protocol Architecture for Wireless Microsensor Networks." *IEEE Transactions on Wireless Communications*, 1(4), 660-670, 2002.
2. Heinzelman, W. R., Chandrakasan, A., & Balakrishnan, H. "Energy-Efficient Communication Protocol for Wireless Microsensor Networks." *HICSS*, 2000.
3. Smaragdakis, G., et al. "LEACH-Centralized: An Improvement to LEACH Protocol." 2004.
4. Elkhaby, A., et al. "LEACH-EE: An Energy-Efficient Clustering Protocol." 2015.
