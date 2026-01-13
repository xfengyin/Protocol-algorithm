# LEACH协议算法

<div align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/build-passing-brightgreen.svg" alt="Build Status">
  <img src="https://img.shields.io/badge/coverage-90%2B-blue.svg" alt="Coverage">
  <img src="https://img.shields.io/badge/docs-available-blue.svg" alt="Documentation">
</div>

## 项目概述

LEACH（Low Energy Adaptive Clustering Hierarchy）是无线传感器网络（WSN）中的一种经典分簇协议算法，旨在通过分布式簇头选择机制延长网络生命周期。本项目实现了LEACH协议算法，并结合AI技术进行了优化，支持多种簇首选择策略。

### 核心思想
- 采用"轮"的概念，周期性地选择簇头
- 随机选择簇头，平衡节点能量消耗
- 基于距离动态调整节点发送功率
- 支持AI辅助的智能簇首选择

## 功能特性

| 功能 | 描述 |
|------|------|
| 经典LEACH算法 | 实现了原始LEACH协议的核心功能 |
| AI辅助簇首选择 | 支持基于机器学习的智能簇首选择 |
| 多种节点分布 | 支持均匀、集中、随机、环形、网格等多种节点分布 |
| 可视化展示 | 提供分簇结果的图形化展示 |
| 模块化设计 | 清晰的代码结构，易于扩展和维护 |
| 完整测试用例 | 覆盖核心功能的单元测试 |
| 高质量训练数据 | 支持生成多样化的WSN训练数据 |

## 安装说明

### 前置要求
- Python 3.10 或更高版本
- uv 包管理工具

### 安装步骤

1. **克隆仓库**
   ```bash
   git clone https://github.com/your-username/Protocol-algorithm.git
   cd Protocol-algorithm
   ```

2. **使用uv安装依赖**
   ```bash
   uv install
   ```

3. **或使用pip安装依赖**
   ```bash
   pip install numpy matplotlib scikit-learn
   ```

## 使用示例

### 基本用法

```python
from src.leach import run

# 运行LEACH算法（传统模式）
run()
```

### AI辅助模式

```python
from src.leach import run

# 运行LEACH算法（AI辅助模式）
run(use_ai=True, model_path='data/cluster_model.joblib')
```

### 生成训练数据

```bash
python generate_data.py
```

### 运行测试

```bash
python test.py
```

## 项目结构

```
Protocol-algorithm/
├── src/                 # 源码目录
│   ├── leach/          # LEACH协议核心实现
│   │   ├── __init__.py
│   │   ├── core.py     # 核心算法实现
│   │   ├── utils.py    # 工具函数
│   │   └── visualization.py # 可视化功能
│   ├── ai/             # AI模型相关
│   │   ├── __init__.py
│   │   ├── model.py    # 模型定义
│   │   ├── trainer.py  # 模型训练
│   │   └── inference.py # 模型推理
│   └── data/           # 数据管理
│       ├── __init__.py
│       ├── generator.py # 数据生成
│       ├── cleaner.py   # 数据清洗
│       └── annotator.py # 数据标注
├── data/               # 生成的数据目录
├── tests/              # 测试用例
├── generate_data.py    # 数据生成脚本
├── test.py             # 测试脚本
├── pyproject.toml      # 项目配置文件
└── README.md           # 项目说明文档
```

## 配置指南

### LEACH算法参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| P | 簇首选择概率 | 0.05 |
| r | 当前轮数 | 0 |
| k | 分簇轮数 | 20 |
| use_ai | 是否使用AI辅助 | False |

### AI模型参数

| 参数 | 描述 | 可选值 | 默认值 |
|------|------|--------|--------|
| model_type | 模型类型 | random_forest, svm, mlp | random_forest |
| n_estimators | 随机森林树数量 | - | 100 |
| max_depth | 树的最大深度 | - | 10 |
| hidden_layer_sizes | MLP隐藏层大小 | - | (128, 64, 32) |

## 贡献指南

我们欢迎社区贡献！如果您想参与本项目，请遵循以下步骤：

1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启一个 Pull Request

### 代码规范

- 遵循PEP 8编码规范
- 使用类型注解
- 为新功能添加测试用例
- 更新相关文档

## 许可证信息

本项目采用MIT许可证。详情请见 [LICENSE](LICENSE) 文件。

## 参考文献

- W. R. Heinzelman, A. Chandrakasan, and H. Balakrishnan, "Energy-efficient communication protocol for wireless microsensor networks," in Proceedings of the 33rd Annual Hawaii International Conference on System Sciences, 2000.
- Wikipedia: [LEACH protocol](https://en.wikipedia.org/wiki/Low_Energy_Adaptive_Clustering_Hierarchy)

## 联系方式

- 项目地址: [https://github.com/your-username/Protocol-algorithm](https://github.com/your-username/Protocol-algorithm)
- 报告问题: [https://github.com/your-username/Protocol-algorithm/issues](https://github.com/your-username/Protocol-algorithm/issues)

---

<div align="center">
  <strong>感谢使用 LEACH 协议算法项目！</strong>
</div>
