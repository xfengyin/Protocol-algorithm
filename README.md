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
pip install -e ".[all]"  # 全部依赖（= 基础依赖 + torch + 测试工具）
pip install -e ".[ai]"   # 只加 AI 依赖（torch）
pip install -e ".[dev]"  # 只加开发工具（pytest / black / isort / mypy）
```

> 依赖清单只保留有真实 import 使用者的包（`requirements.txt` 顶部写了判据与删除记录）。
> `jupyter` / `ipykernel` / `tqdm` / `simpy` / `shap` / `seaborn` / `lightgbm` 于 2026-10-15 删除，
> 原先的 `.[simulation]` extra 也一并删除（它唯一的成员就是没被 import 的 `simpy`）。
> 图形/动画/并行仿真所需的库都在基础依赖里（numpy / pandas / matplotlib）。

## 快速开始

```python
from src.models.network import Network
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

# 运行仿真：协议实现按 protocol_name 从 LEACHRegistry 取（"leach" -> ClassicLEACH(p=0.05)），
# 不需要先造一个协议实例；要换簇头概率就改注册表里的实现，见 examples/advanced_usage.py
results = network.simulate_network(rounds=5000, protocol_name="leach")

# 查看结果
print(f"网络生命周期: {results['network_lifetime']} 轮")
print(f"最终剩余能量: {results['final_energy']:.4f} J")
# 注意 results['total_energy'] 是「每轮」剩余能量的列表，不是标量：末轮总能量 = final_energy
print(f"逐轮记录条数: {len(results['total_energy'])}")
```

## 命令行使用

```bash
# 最小示例：main.py 不接收任何命令行参数（100 节点 / 1000 轮 / 种子 42，硬编码在文件里）
python main.py

# 完整命令行接口在 cli.py：--config --output --protocol --rounds --nodes --seed --no-plot
python cli.py --config config/config.yaml --output results/leach/
python cli.py --protocol leach_ee --rounds 5000 --nodes 100   # 协议名用下划线形式
python cli.py --rounds 1000 --no-plot                        # 不画图，只存 results.json

# 注意 config/leach_ai.yaml 里的 protocol: leach_ai 目前没有注册进 LEACHRegistry，
# `python cli.py --config config/leach_ai.yaml` 会报 Unknown protocol（可用
# python -c "from src.leach.variants import LEACHRegistry; print(LEACHRegistry.list_protocols())"
# 查看已注册的协议名）。AI 簇头选择的可用入口是 examples/run_ai_leach.py。

# 对比实验
python examples/compare_variants.py --output results/comparison/
```

## 示例脚本

`examples/` 下的每个脚本都可以直接 `python examples/<name>.py` 运行（脚本自己把项目根加进 `sys.path`）：

| 脚本 | 行数 | 演示内容 |
|------|------|----------|
| `run_classic_leach.py` | 177 | 最小可用路径：建网 -> 跑经典 LEACH -> 画指标图 -> 存结果，适合第一次上手 |
| `run_ai_leach.py` | 369 | 用经典 LEACH 生成训练集 -> 训练 sklearn 选择器 -> 手动逐轮用模型选簇头，并与经典 LEACH 对比（不依赖注册表里存在 AI 协议） |
| `compare_variants.py` | 387 | 四种变体（LEACH / C / EE / M）多次重复实验 -> 汇总统计、排名、对比图表 |
| `reproduce.py` | 488 | 复现 Heinzelman 2000 论文：生命周期曲线、能量消耗趋势、簇头概率 p 参数扫描 + Markdown 报告 |
| `advanced_usage.py` | 608 | 高级用法：并行仿真、参数扫描、自定义协议注册、批量实验、拓扑动画、能量模型调参 |

`examples/` 与 `main.py` / `cli.py` 都在 CI 的 lint（black / isort / flake8）与 mypy 范围内。

## 项目结构

```
Protocol-algorithm/
├── src/
│   ├── leach/          # LEACH 协议变体（classic / c / ee / m + 注册表 LEACHRegistry）
│   ├── models/         # 数据模型（node / cluster_head / base_station / network）
│   ├── energy/         # 能量模型（含 radio_model.py 兼容层）
│   ├── ai/             # AI 优化（选择器基类 + sklearn / PyTorch 实现 + trainer）
│   ├── simulation/     # 仿真引擎（并行 / 批量 / 参数扫描）
│   ├── visualization/  # 可视化（指标图 / 对比图 / 动画）
│   ├── data/           # 训练数据生成与采样
│   └── config/         # 配置校验（FullConfig.from_yaml；当前 CLI 未使用，见 CONTRIBUTING）
├── config/             # 运行配置（config.yaml / leach_ai.yaml）
├── examples/           # 5 个可直接运行的示例（见上一节表格）
├── tests/              # 单元测试（pytest）
├── docs/               # 教程 / 算法说明 / API 参考
├── main.py             # 零参数最小示例
├── cli.py              # 命令行接口
├── requirements.txt    # CI 安装的依赖清单
├── pyproject.toml      # 打包元数据 / extras / black / isort / mypy 配置
├── .flake8             # flake8 规则（唯一真相源，CI 与 pre-commit 都读它）
└── .gitea/workflows/ci.yml  # CI 门禁定义
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

**集成方式**: 在 `src/ai/` 目录实现了特征工程（`feature_engineering.py`）与训练器（`trainer.py`）；数据集与评估指标本身没有落地（README 的未来扩展里保留了 WSN-DS 数据集一项）

### 3. WSN-Scheduling-with-Reinforcement-Learning
**GitHub**: https://github.com/AliAmini93/WSN-Scheduling-with-Reinforcement-Learning

**借鉴内容**:
- DQN强化学习算法思想
- Stable Baselines3集成
- 智能调度优化策略

**集成方式**: 在 `src/ai/` 目录预留了强化学习接口，便于未来扩展

### 4. 其他项目借鉴

- **Slipstream-Max/wsn**: 简单直观的可视化设计
- **Nachi28/WSN_LEACH**: 结果保存思路（本项目落成 `results.json` + 图表，**没有** Excel/xlsx 导出）
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

## 质量门禁（CI）

`.gitea/workflows/ci.yml` 里每个检查都是硬门禁：**能红也能绿**。判据与负对照（证明这个检查
自己会报错，而不是永远沉默）都写在 workflow 里：

| 检查 | 判据 | 负对照 |
|------|------|--------|
| black / isort / flake8 | `src/ tests/ examples/ main.py cli.py __init__.py` 全部无差异；规则只有 `.flake8` 一份 | —— （格式门禁靠 diff 自证） |
| mypy | 上述范围 **0 错误**（改前是 `continue-on-error: true` + 41 个存量错误） | 植入必然出错的类型赋值，断言 mypy 非零退出 |
| bandit | `src/` 上 `--severity-level=medium --confidence-level=medium` 无未登记告警 | 植入 `eval()` / `shell=True`，断言非零退出 |
| pip-audit | 审计 CI 真正装上的依赖集，无未登记漏洞（登记用 `--ignore-vuln <ID>`） | 审计 `pyyaml==5.3.1`，断言非零退出 |
| pytest | Python 3.11 / 3.12 两个矩阵腿全绿 | —— （49 个用例本身就带断言） |
| 覆盖率 | `--cov-fail-under=44`：存量实测 1930 行覆盖 858 行 = 44.46%，阈值取 floor 值 | 单跑一个测试文件 + 阈值 90，断言因「覆盖率未达标」而失败 |

覆盖率是**棘轮**（ratchet）：44% 是当前存量的地板，不是达标线 —— 新增未测代码会让 job 变红，
要加功能就得同时加测试。`data/`、`visualization/` 目前几乎无测试覆盖，
抬高阈值需要先补测试（见 CONTRIBUTING 的「覆盖率要求」）。

## License

MIT License
