# WSN LEACH 协议仿真项目 - 贡献指南

感谢你对本项目感兴趣！我们欢迎所有形式的贡献，包括代码、文档、测试、问题报告和功能建议。
在开始贡献之前，请仔细阅读本指南。

---

## 目录

- [行为准则](#行为准则)
- [如何贡献代码](#如何贡献代码)
  - [Fork & Pull Request 流程](#fork--pull-request-流程)
  - [分支策略](#分支策略)
- [开发环境设置](#开发环境设置)
  - [前置要求](#前置要求)
  - [安装步骤](#安装步骤)
  - [虚拟环境](#虚拟环境)
  - [安装开发依赖](#安装开发依赖)
- [编码规范](#编码规范)
  - [PEP 8 与代码格式化](#pep-8-与代码格式化)
  - [命名规范](#命名规范)
  - [类型提示](#类型提示)
  - [导入规范](#导入规范)
  - [文档字符串](#文档字符串)
- [代码提交规范](#代码提交规范)
  - [Conventional Commits](#conventional-commits)
  - [提交格式示例](#提交格式示例)
  - [使用 pre-commit 钩子](#使用-pre-commit-钩子)
- [测试要求](#测试要求)
  - [运行测试](#运行测试)
  - [覆盖率要求](#覆盖率要求)
  - [编写测试用例](#编写测试用例)
  - [基准测试](#基准测试)
- [代码审查流程](#代码审查流程)
  - [审查清单](#审查清单)
  - [审查标准](#审查标准)
- [如何报告 Bug](#如何报告-bug)
- [如何提出新功能](#如何提出新功能)
- [架构设计原则](#架构设计原则)
- [项目结构](#项目结构)
- [发布流程](#发布流程)
- [联系方式](#联系方式)

---

## 行为准则

本项目遵循开源社区行为准则。请保持友善、尊重和专业，对所有参与者一视同仁。

---

## 如何贡献代码

### Fork & Pull Request 流程

1. **Fork 本仓库**

   点击 GitHub 页面右上角的 "Fork" 按钮，将仓库复制到你自己的账号下。

2. **克隆 Fork 的仓库**

   ```bash
   git clone https://github.com/YOUR_USERNAME/protocol-algorithm.git
   cd protocol-algorithm
   ```

3. **添加上游远程仓库**

   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/protocol-algorithm.git
   git fetch upstream
   ```

4. **创建功能分支**

   ```bash
   git checkout main
   git pull upstream main
   git checkout -b feature/your-feature-name
   ```

   分支命名约定：
   - `feature/xxx` — 新功能
   - `bugfix/xxx` — Bug 修复
   - `hotfix/xxx` — 紧急修复
   - `docs/xxx` — 文档更新
   - `refactor/xxx` — 重构
   - `test/xxx` — 测试相关

5. **进行开发并提交**

   ```bash
   # 编辑代码...

   # 格式化代码
   black src/ tests/
   isort src/ tests/

   # 运行测试
   pytest tests/ -v --cov=src --cov-report=term-missing

   # 提交
   git add .
   git commit -m "feat(leach): 添加 LEACH-F 能量感知变体"
   ```

6. **推送到远程**

   ```bash
   git push origin feature/your-feature-name
   ```

7. **创建 Pull Request**

   - 前往 GitHub，点击 "Compare & pull request"
   - 填写 PR 模板，描述变更内容、测试情况和关联 Issue
   - 等待代码审查

8. **根据审查意见修改**

   - 在本地修改后，追加提交到同一分支
   - 推送后 PR 会自动更新
   - 重复审查-修改流程直到合入

### 分支策略

```
main                ← 稳定发布分支
  └── develop       ← 集成开发分支
       ├── feature/xxx
       ├── bugfix/xxx
       └── docs/xxx
```

- `main`：仅接受来自 `develop` 的合并，用于正式发布
- `develop`：日常开发集成分支
- 功能分支从 `develop` 切出，完成后合并回 `develop`

---

## 开发环境设置

### 前置要求

- **Python**: >= 3.9（推荐 3.10 或更高）
- **Git**: >= 2.30
- **pip**: >= 21.0
- **操作系统**: Linux / macOS / Windows (WSL2 推荐)

### 安装步骤

1. 克隆仓库后，进入项目根目录：

   ```bash
   cd protocol-algorithm
   ```

2. 验证 Python 版本：

   ```bash
   python --version
   # 应输出 Python 3.9.x 或更高版本
   ```

### 虚拟环境

**推荐使用 venv（Python 内置）**：

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Linux / macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# 升级 pip
pip install --upgrade pip
```

**或使用 conda**：

```bash
conda create -n leach-sim python=3.10
conda activate leach-sim
```

**或使用 Poetry**：

```bash
pip install poetry
poetry install
```

### 安装开发依赖

```bash
# 安装项目全部依赖（仿真 + AI + 开发工具）
pip install -e ".[all,dev]"

# 仅安装仿真依赖
pip install -e ".[simulation]"

# 仅安装 AI 依赖
pip install -e ".[ai]"

# 安装开发工具（格式化、类型检查、测试）
pip install -e ".[dev]"
```

### 验证安装

```bash
# 运行测试
pytest tests/ -v

# 运行示例
python examples/run_classic_leach.py
```

---

## 编码规范

本项目严格遵循 **PEP 8** 编码规范，并使用自动化工具保证一致性。

### PEP 8 与代码格式化

**使用 Black 进行格式化**（强制风格）：

```bash
# 格式化所有代码
black src/ tests/ examples/

# 检查格式（不修改）
black --check src/ tests/

# 指定行长度
black --line-length 100 src/
```

**使用 isort 整理导入**：

```bash
# 排序导入
isort src/ tests/

# 检查导入排序
isort --check src/ tests/
```

**使用 flake8 进行规范检查**：

```bash
flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W503
```

**使用 mypy 进行类型检查**：

```bash
mypy src/ --ignore-missing-imports
```

### 命名规范

| 类型 | 风格 | 示例 |
|------|------|------|
| 模块/包名 | `snake_case` | `radio_model`, `energy` |
| 类名 | `PascalCase` | `ClusterHead`, `Network` |
| 函数/方法/变量名 | `snake_case` | `calculate_energy`, `node_id` |
| 常量名 | `UPPER_SNAKE_CASE` | `DEFAULT_ROUNDS`, `MAX_NODES` |
| 私有属性 | 前缀 `_` | `_internal_cache` |

示例：

```python
# 正确
class FirstOrderRadioModel:
    """一阶无线电能量模型"""

    DEFAULT_ETX = 50e-9  # 常量：全大写 + 下划线
    MAX_TRANSMIT_DISTANCE = 100.0

    def __init__(self, etx: float = DEFAULT_ETX) -> None:
        self.etx = etx
        self._energy_map: dict[int, float] = {}  # 私有属性

    def calculate_tx_energy(self, distance: float, packet_size: int) -> float:
        """计算发送能耗"""
        if distance <= self.MAX_TRANSMIT_DISTANCE:
            return self.etx * packet_size
        return self._calculate_far_field(distance, packet_size)

    def _calculate_far_field(self, distance: float, packet_size: int) -> float:
        """远场能耗计算（私有方法）"""
        # ...
```

### 类型提示

**所有公共函数和方法必须添加类型提示**：

```python
from typing import Optional


def select_cluster_heads(
    nodes: list[dict],
    p: float,
    current_round: int,
    threshold: Optional[float] = None,
) -> list[int]:
    """
    选择本轮簇头节点

    Args:
        nodes: 节点列表
        p: 期望簇头比例
        current_round: 当前轮次
        threshold: 选择阈值，默认根据 LEACH 公式计算

    Returns:
        被选中的簇头节点 ID 列表
    """
    # 实现...
```

**复杂类型使用 typing 模块**：

```python
from typing import NamedTuple
from dataclasses import dataclass


@dataclass
class SimulationResult:
    """仿真结果数据结构"""

    round_number: int
    alive_nodes: int
    total_energy: float
    cluster_heads: list[int]


class NetworkMetrics(NamedTuple):
    """网络性能指标"""

    network_lifetime: int
    total_energy_consumed: float
    packets_to_bs: int
    average_energy_per_node: float
```

### 导入规范

导入顺序遵循：**标准库 → 第三方库 → 项目内部库**，各组之间空一行：

```python
# 1. 标准库
import os
import logging
from typing import Optional

# 2. 第三方库
import numpy as np
import matplotlib.pyplot as plt

# 3. 项目内部库
from src.energy.radio_model import FirstOrderRadioModel
from src.models.node import Node
from src.config.validator import ConfigValidator
```

每个 `import` 语句只导入一个模块。

### 文档字符串

**所有公共模块、类、函数必须编写 Google 风格的文档字符串**：

```python
class LEACHProtocol:
    """
    LEACH 协议基类

    实现 LEACH 协议的核心逻辑，包括分簇、簇头选举和数据传输阶段。
    子类应继承此类并重写特定方法实现不同的 LEACH 变体。

    Attributes:
        network: 网络拓扑对象
        config: 协议配置字典
        current_round: 当前轮次

    Example:
        >>> network = Network(n_nodes=100)
        >>> leach = LEACHProtocol(network)
        >>> results = leach.run(5000)
    """

    def calculate_threshold(
        self,
        node_id: int,
        current_round: int,
        total_rounds: int,
        cluster_probability: float,
        has_been_cluster_head: bool,
    ) -> float:
        """
        计算节点成为簇头的阈值

        根据 LEACH 协议公式计算节点在当前轮次被选为簇头的概率阈值。
        如果节点在最近 1/p 轮内已经担任过簇头，则阈值为 0。

        Args:
            node_id: 节点唯一标识
            current_round: 当前仿真轮次（从 0 开始）
            total_rounds: 总仿真轮数
            cluster_probability: 期望簇头比例 p
            has_been_cluster_head: 节点是否已在最近周期内担任过簇头

        Returns:
            阈值 T(n)，范围 [0, 1]。返回值越大，节点越可能成为簇头

        Raises:
            ValueError: 当 cluster_probability <= 0 或 > 1 时抛出

        See Also:
            :meth:`select_cluster_heads`: 簇头选择主方法
        """
        # 实现...
```

---

## 代码提交规范

### Conventional Commits

本项目遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范。
所有提交必须按照以下格式编写：

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### 提交类型

| 类型 | 说明 | 示例 |
|------|------|------|
| `feat` | 新功能 | `feat(leach): 添加 LEACH-F 变体` |
| `fix` | Bug 修复 | `fix(energy): 修复远场能耗计算溢出` |
| `docs` | 文档更新 | `docs(readme): 添加安装说明` |
| `style` | 代码格式（不影响逻辑） | `style: 使用 black 格式化` |
| `refactor` | 重构（既不是新功能也不是修复） | `refactor(ai): 将选择器抽象为接口` |
| `perf` | 性能优化 | `perf(simulation): 使用 NumPy 批量计算` |
| `test` | 测试相关 | `test(leach): 添加簇头分布测试` |
| `chore` | 构建/工具/配置变更 | `chore(ci): 添加覆盖率上传` |
| `ci` | CI 配置变更 | `ci(github): 更新工作流` |

### 提交格式示例

```bash
# 功能提交
git commit -m "feat(ai): 添加 LightGBM 簇头选择器"

# Bug 修复提交（关联 Issue）
git commit -m "fix(energy): 修复双径模型计算中的除以零错误

当节点与基站重合时距离为 0，导致路径损耗计算异常。
添加最小距离阈值 0.1m 避免除以零。

Fixes #42"

# 重构提交
git commit -m "refactor(leach): 将阈值计算提取为独立策略类

遵循开闭原则，新增 LEACH 变体无需修改主流程。
新增 ThresholdStrategy 抽象基类，各变体实现各自策略。"

# 包含 BREAKING CHANGE 的提交
git commit -m "feat(simulation)!: 重构仿真引擎为事件驱动

BREAKING CHANGE: 仿真引擎 API 变更，run() 方法改为
simulate() 并使用 simpy 环境。旧 API 将在 v2.0 移除。

Migration guide:
- network.run(rounds) → network.simulate(rounds)
- 结果返回类型从 dict 改为 SimulationResult
"
```

### 使用 pre-commit 钩子

推荐使用 pre-commit 在提交前自动检查：

1. 安装 pre-commit：

   ```bash
   pip install pre-commit
   pre-commit install
   ```

2. 提交时会自动运行以下检查：
   - Black 格式化
   - isort 导入排序
   - flake8 规范检查
   - mypy 类型检查（可选）

---

## 测试要求

### 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试文件
pytest tests/test_energy.py -v

# 运行特定测试方法
pytest tests/test_energy.py::test_first_order_radio -v

# 运行带覆盖率报告
pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

# 运行并生成覆盖率 XML（用于 CI）
pytest tests/ --cov=src --cov-report=xml

# 并行加速测试
pytest tests/ -n auto
```

### 覆盖率要求

| 模块类型 | 最低覆盖率 | 说明 |
|----------|-----------|------|
| 核心仿真逻辑 | >= 90% | `src/leach/`, `src/simulation/` |
| 能量模型 | >= 90% | `src/energy/` |
| AI 优化模块 | >= 80% | `src/ai/` |
| 可视化模块 | >= 70% | `src/visualization/` |
| 工具类 | >= 80% | `src/utils/`, `src/config/` |

**整体项目覆盖率目标：>= 80%**

### 编写测试用例

**单元测试示例**：

```python
import pytest
import numpy as np

from src.energy.radio_model import FirstOrderRadioModel
from src.leach.classic import ClassicLEACH
from src.models.network import Network


class TestFirstOrderRadioModel:
    """一阶无线电能量模型测试"""

    @pytest.fixture
    def radio_model(self) -> FirstOrderRadioModel:
        return FirstOrderRadioModel()

    def test_tx_energy_short_distance(self, radio_model: FirstOrderRadioModel) -> None:
        """测试短距离发送能耗计算"""
        distance = 10.0  # 小于阈值距离
        packet_size = 4000  # bits

        energy = radio_model.calculate_tx_energy(distance, packet_size)

        assert energy > 0
        assert energy == pytest.approx(radio_model.etx * packet_size)

    def test_tx_energy_zero_distance_raises(self, radio_model: FirstOrderRadioModel) -> None:
        """测试零距离发送应抛出异常或使用最小距离"""
        with pytest.raises(ValueError, match="distance must be positive"):
            radio_model.calculate_tx_energy(0.0, 4000)

    @pytest.mark.parametrize(
        "distance,expected_multiplier",
        [
            (10.0, 1.0),  # 自由空间
            (50.0, 1.0),  # 自由空间
            (100.0, 4.0),  # 多径衰减（距离翻倍，能耗 x4）
        ],
    )
    def test_tx_energy_with_distance(
        self,
        radio_model: FirstOrderRadioModel,
        distance: float,
        expected_multiplier: float,
    ) -> None:
        """参数化测试不同距离下的能耗"""
        base_energy = radio_model.etx * 4000
        energy = radio_model.calculate_tx_energy(distance, 4000)

        assert energy == pytest.approx(base_energy * expected_multiplier, rel=0.1)


class TestClassicLEACH:
    """经典 LEACH 协议测试"""

    @pytest.fixture
    def network(self) -> Network:
        return Network(n_nodes=100, area=(0, 100, 0, 100))

    @pytest.fixture
    def leach(self, network: Network) -> ClassicLEACH:
        return ClassicLEACH(network, cluster_probability=0.05)

    def test_cluster_head_selection_count(self, leach: ClassicLEACH) -> None:
        """测试簇头数量在期望范围内"""
        cluster_heads = leach.select_cluster_heads(current_round=0)

        expected_count = 100 * 0.05  # p = 0.05
        # 允许 50% 波动（随机算法）
        assert len(cluster_heads) >= expected_count * 0.5
        assert len(cluster_heads) <= expected_count * 2.0

    def test_threshold_calculation(self, leach: ClassicLEACH) -> None:
        """测试阈值计算符合 LEACH 公式"""
        threshold = leach.calculate_threshold(
            node_id=0,
            current_round=0,
            cluster_probability=0.05,
        )

        expected = 0.05 / (1 - 0.05 * 0)  # T(n) = p / (1 - p * r)
        assert threshold == pytest.approx(expected, rel=1e-9)
```

**测试命名规范**：

- 测试文件：`test_<module_name>.py`
- 测试类：`Test<ClassName>`
- 测试方法：`test_<method_name>_<scenario>`
- 参数化：使用 `@pytest.mark.parametrize`

**测试标记**：

```python
@pytest.mark.slow  # 标记慢速测试
@pytest.mark.ai  # 标记 AI 相关测试
@pytest.mark.parametrize("p", [0.01, 0.05, 0.1])
def test_cluster_probability_impact(p: float) -> None:
    ...
```

运行慢速测试：

```bash
pytest tests/ -m slow  # 仅运行慢速测试
pytest tests/ -m "not slow"  # 跳过慢速测试
```

### 基准测试

对于性能相关变更，需要运行基准测试：

```bash
# 运行基准测试
python tests/test_benchmarks.py

# 对比性能
python -m timeit -n 100 "from src.leach.classic import ClassicLEACH; ..."
```

---

## 代码审查流程

### 审查清单

提交 PR 前，请自查以下清单：

- [ ] 代码遵循 PEP 8 和 Black 格式化
- [ ] 所有公共函数添加了类型提示
- [ ] 所有公共函数添加了文档字符串
- [ ] 新增功能包含单元测试
- [ ] 测试覆盖率 >= 80%
- [ ] 所有测试通过（`pytest tests/ -v`）
- [ ] 提交信息遵循 Conventional Commits
- [ ] PR 描述清晰，包含变更说明和测试情况
- [ ] 关联了相关 Issue（如有）
- [ ] 更新了相关文档（如 README、CHANGELOG）
- [ ] 没有引入新的安全漏洞或隐私泄露
- [ ] 代码不包含调试代码（如 `print()`, `pdb`）

### 审查标准

**PR 合入需要**：

1. **至少 1 名维护者批准**（核心模块需 2 名）
2. **CI 全部通过**（测试、格式化、类型检查）
3. **无未解决的审查意见**
4. **无冲突**

**审查关注点**：

- **正确性**：逻辑是否正确？边界条件是否处理？
- **可读性**：代码是否清晰？命名是否合理？
- **可维护性**：是否遵循 SOLID 原则？是否有重复代码？
- **性能**：是否有不必要的计算或内存分配？
- **安全性**：是否有注入风险？敏感数据是否处理得当？
- **测试**：测试是否充分？是否覆盖了异常路径？

---

## 如何报告 Bug

使用 GitHub Issues 报告 Bug，请遵循以下模板：

### Bug 报告模板

```markdown
## Bug 描述

简要描述遇到的问题。

## 复现步骤

1. 运行 `python examples/run_classic_leach.py`
2. 使用配置 `config/config.yaml`
3. 观察...

## 期望行为

应该发生什么。

## 实际行为

实际发生了什么。

## 环境信息

- Python 版本：3.x.x
- 操作系统：Linux / macOS / Windows
- 项目版本：v1.0.0（或 commit hash）

## 日志/截图

```
粘贴相关错误日志或截图
```

## 其他信息

任何其他有助于定位问题的信息。
```

### Bug 严重程度分类

| 级别 | 说明 | 响应时间 |
|------|------|----------|
| P0 - 致命 | 系统崩溃、数据损坏 | 24 小时内 |
| P1 - 严重 | 核心功能不可用 | 3 天内 |
| P2 - 一般 | 部分功能异常，有替代方案 | 1 周内 |
| P3 - 轻微 | UI 问题、拼写错误 | 按优先级排期 |

---

## 如何提出新功能

### 功能请求模板

```markdown
## 功能描述

简要描述期望的新功能。

## 使用场景

描述这个功能能解决什么问题，举具体例子。

## 建议实现

如果有技术方案想法，可以在这里描述。

## 替代方案

是否考虑过其他实现方式？

## 额外信息

- 是否有相关论文或参考资料？
- 是否有类似功能的开源项目可以参考？
```

### 功能开发流程

1. **创建 Feature Request Issue**，描述功能需求
2. **与维护者讨论**，确定技术方案
3. **Fork 仓库**，创建功能分支
4. **开发并编写测试**
5. **提交 PR**，关联 Issue
6. **代码审查**
7. **合入 main/develop**

---

## 架构设计原则

本项目遵循以下设计原则，贡献代码时请务必遵守：

### SOLID 原则

| 原则 | 说明 | 示例 |
|------|------|------|
| **单一职责** | 每个类/模块只负责一项职责 | `FirstOrderRadioModel` 只计算能量 |
| **开闭原则** | 对扩展开放，对修改关闭 | 新增 LEACH 变体继承基类 |
| **里氏替换** | 子类可替换父类 | 所有 LEACH 变体可互换 |
| **接口隔离** | 接口尽量小而专一 | 拆分 `EnergyModel` 和 `RadioModel` |
| **依赖倒置** | 依赖抽象而非具体实现 | 面向 `BaseLEACH` 编程 |

### 插件化架构

新增 LEACH 变体只需：

1. 继承 `BaseLEACH` 类
2. 实现 `select_cluster_heads()` 和 `calculate_threshold()` 方法
3. 在 `config.yaml` 中注册

```python
from src.leach.base import BaseLEACH


class LEACHF(BaseLEACH):
    """LEACH-F：考虑能量因子的 LEACH 变体"""

    def select_cluster_heads(self, current_round: int) -> list[int]:
        # 实现...
        pass

    def calculate_threshold(
        self,
        node_id: int,
        current_round: int,
        cluster_probability: float,
    ) -> float:
        # 实现...
        pass
```

---

## 项目结构

```
protocol-algorithm/
├── src/                      # 源代码
│   ├── leach/               # LEACH 协议变体
│   │   ├── base.py          # 基类
│   │   ├── classic.py       # 经典 LEACH
│   │   ├── leach_c.py       # LEACH-C
│   │   ├── leach_ee.py      # LEACH-EE
│   │   └── leach_m.py       # LEACH-M
│   ├── models/              # 数据模型
│   │   ├── node.py          # 节点
│   │   ├── network.py       # 网络
│   │   └── base_station.py  # 基站
│   ├── energy/              # 能量模型
│   │   └── radio_model.py   # 一阶无线电模型
│   ├── ai/                  # AI 优化模块
│   │   ├── selector.py      # 选择器接口
│   │   ├── sklearn_selector.py
│   │   └── pytorch_selector.py
│   ├── simulation/          # 仿真引擎
│   │   └── engine.py
│   ├── visualization/       # 可视化
│   │   └── metrics_plots.py
│   ├── config/              # 配置管理
│   │   └── validator.py
│   └── data/                # 数据生成
│       └── generator.py
├── tests/                    # 测试代码
├── config/                   # 配置文件
├── examples/                 # 示例脚本
├── docs/                     # 文档
├── results/                  # 仿真结果输出
├── pyproject.toml           # 项目配置
└── requirements.txt         # 依赖列表
```

---

## 发布流程

维护者负责发布新版本：

1. **更新版本号**（`pyproject.toml`）
2. **更新 CHANGELOG.md**
3. **创建 Git Tag**
4. **发布 GitHub Release**

```bash
# 打标签
git tag -a v1.1.0 -m "Release v1.1.0: 添加 LEACH-F 变体"
git push origin v1.1.0
```

---

## 联系方式

- **GitHub Issues**: [报告 Bug / 提出功能](https://github.com/xfengyin/protocol-algorithm/issues)
- **讨论区**: [GitHub Discussions](https://github.com/xfengyin/protocol-algorithm/discussions)
- **邮件**: [项目维护者邮箱]

---

## 许可证

本项目采用 MIT 许可证。提交代码即表示你同意将代码以 MIT 许可证发布。

---

**感谢你的贡献！🎉**
