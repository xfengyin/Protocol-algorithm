# LEACH 协议算法原理

## 1. 概述

LEACH (Low-Energy Adaptive Clustering Hierarchy) 是一种专为无线传感器网络设计的分层路由协议。其主要目标是延长网络寿命，通过动态选择簇头来实现能量均衡。

## 2. 网络模型

### 2.1 First Order Radio Model

LEACH 使用 First Order Radio Model 来计算能量消耗：

```
发送能耗:
  E_tx(d, k) = E_elec * k + epsilon_fs * k * d^2,  if d <= d_threshold
             = E_elec * k + epsilon_mp * k * d^4,  if d > d_threshold

接收能耗:
  E_rx(k) = E_elec * k

数据聚合能耗:
  E_da(k) = E_da * k
```

### 2.2 参数说明

| 参数 | 描述 | 默认值 |
|------|------|--------|
| E_elec | 发射/接收电路能耗 | 50 nJ/bit |
| epsilon_fs | 自由空间放大器系数 | 10 pJ/bit/m² |
| epsilon_mp | 多径衰落放大器系数 | 0.0013 pJ/bit/m⁴ |
| d_threshold | 距离阈值 | 87 m |
| E_da | 数据聚合能耗 | 5 nJ/bit |

## 3. LEACH 协议机制

### 3.1 轮结构

LEACH 以"轮"为单位运行，每轮分为两个阶段：

1. **设置阶段 (Setup Phase)**
   - 簇头选举
   - 簇形成
   - 调度创建

2. **稳定阶段 (Steady Phase)**
   - 数据传输
   - 多路复用

### 3.2 簇头选择

每个节点根据阈值函数决定是否成为簇头：

```
T(n) = p / (1 - p * (r mod 1/p)),  if n ∈ G
T(n) = 0,                         otherwise
```

其中：
- p = 期望的簇头比例 (默认 0.05)
- r = 当前轮数
- G = 过去 1/p 轮中未成为簇头的节点集合

### 3.3 簇形成

非簇头节点根据信号强度加入最近的簇头，形成 TDMA 调度。

## 4. LEACH 变体

### 4.1 LEACH-C (Centralized)

由基站集中选择簇头，基于全局能量和位置信息优化。

### 4.2 LEACH-EE (Energy-Efficient)

考虑节点剩余能量和邻居密度，动态调整簇头概率。

### 4.3 LEACH-M (Mobile)

支持移动节点，考虑节点移动性对网络的影响。

## 5. 性能指标

- **网络生命周期**: 第一个节点死亡时的轮数
- **半网络生命周期**: 一半节点死亡时的轮数
- **总能耗**: 网络总能量消耗
- **能耗均衡**: 节点能量方差

## 6. 参考

1. Heinzelman, W. B., et al. "An application-specific protocol architecture for wireless microsensor networks." IEEE Transactions on wireless communications (2002).
