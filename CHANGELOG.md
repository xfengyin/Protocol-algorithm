# Changelog

## [Unreleased]

（两轮未发布改动：2026-10-14 的代码修复轮 + 2026-10-15 的门禁加固轮）

### Fixed (2026-10-14)
- `models/network.py`: 删除 `self._spatial_index.cKDTree__repr__(positions)`
  —— 该方法并不存在（运行时必 AttributeError），且结果变量从未被使用
- `data/sampler.py`: `undersample()` 补上缺失的 `y_pos = y[pos_mask]`
  —— 原代码在 `np.concatenate([y_pos, y_neg])` 处引用了未定义的名字，一调用就 NameError
  （该行被误放在了 `oversample()`，而那里并不需要它）
- `models/network.py`: `stop_condition` 的注解由内置函数 `callable` 改为 `Callable[["Network", int], bool]`
- `ai/feature_engineering.py`: 邻居缓存类型由 `List[int]` 修正为 `List[Node]`（实际存的是节点列表）
- `simulation/engine.py`: 不再把 `str` 类型的 `output_path` 参数重新赋值为 `Path`

### Changed (2026-10-14)
- 全量 `black` + `isort` 格式化（30 个文件）
- 删除 46 处未使用 import、4 处赋值后未使用的局部变量、2 处重复 import
- 新增 `.flake8`，`max-line-length = 100` 与 `[tool.black]` 对齐
  （此前 CI 用 88、black 用 100，同一份代码在两套标准下互相打架）
- `requires-python` 由 `>=3.9` 改为 `>=3.11`：`matplotlib>=3.11.2` 要求 Python>=3.11，与原先声明的 3.9 自相矛盾，导致 3.9/3.10 的测试永远卡在安装依赖
- CI 测试矩阵由 `[3.9, 3.10, 3.11]` 改为 `[3.11, 3.12]`
- mypy 增加 `--explicit-package-bases`；存量类型错误暂以 `continue-on-error` 暴露而不阻塞
  （当时注释写的是 47，**实测为 41 个 / 13 个文件**；已于 2026-10-15 清零并改成硬门禁）

### 门禁加固 (2026-10-15)

主题：把「有门禁但永远不可能红」变成「能红也能绿」。每个检查都补了**负对照** ——
先证明这个检查自己会报错（空输出不能自证清白），基线数据都写在 workflow 注释里。

#### P1 安全门禁：bandit / safety
- `bandit ... || true` 与 `safety check ... || true` 是两条最该拦人的门禁，但扫出任何问题 job 依然绿
- bandit 变成硬门禁；存量实测只有 1 条告警（`src/ai/sklearn_selector.py` 的 B301 pickle），
  用带理由的 `# nosec B301` 登记而不是放行整条规则；新增 `Verify bandit gate can fail` 负对照
- `safety` 在 CI 里**从未真正执行过**：`AttributeError: module 'typer' has no attribute 'rich_utils'`
  （safety 3.1.0 与 typer 0.27.3 不兼容），且 `safety check` 不带目标时审计的是它自己的 venv
- 改用 `pip-audit` 审计测试 job 真正安装的依赖集；新增依赖审计负对照（pyyaml==5.3.1 已知 2 条漏洞）

#### P2 mypy：从 `continue-on-error` 到 0 错误硬门禁
- 实测存量 **41 个错误 / 13 个文件**（注释里的 47 已过时），全部清零
- 根因集中在「声明返回 float/dict 却返回 Any 表达式」（job 不装 numpy/torch/sklearn，
  `--ignore-missing-imports` 让它们退化成 Any）以及选择器基类把 `model` 定死成 NoneType
- 覆盖范围从 `src/` 扩到 `src/ tests/ examples/ main.py cli.py __init__.py`（47 个文件，0 错误）
- 补 `types-PyYAML` 存根（否则 `src/config/validator.py` 报 import-untyped）

#### P3 lint 与类型检查范围
- `examples/`（2029 行，5 个脚本）与 `main.py` / `cli.py` 此前完全不在 lint 范围内
- 规则收敛到 `.flake8` 一份（CI 命令行不再重复 max-line-length / extend-ignore / per-file-ignores）
- `.pre-commit-config.yaml` 与 CI 对齐（此前 flake8 传 88 而 .flake8/black 是 100；
  mypy 多了 `--disallow-untyped-defs` 且缺 `--explicit-package-bases`，hook 直接报错退出；
  bandit 用默认 low severity，会拦下 CI 明确放行的 B403）
- 修掉 examples/ 与根脚本暴露的问题：14 个未使用 import、7 个无占位符 f-string、
  `main.py` 的死变量 `protocol = ClassicLEACH(p=0.05)`（仿真走注册表，这个实例从未被使用）

#### P4 覆盖率
- 阈值不是拍的：存量实测 **1930 行覆盖 858 行 = 44.46%**（CI 上传的 coverage.xml，
  run 10185 与 run 10191 两次独立运行相同），取 floor 值 `--cov-fail-under=44`，
  容差 0.46 个百分点 ≈ 9 条语句；语义是棘轮（存量不许下降），不是达标线
- 新增阈值型负对照：单跑一个测试文件 + 阈值 90，断言「因覆盖率未达标而失败」
  （而不是把真实测试失败误当成门禁生效）

#### X1 CI 不可达分支
- 删除 10 处 `github.server_url` 判断出来的 GitHub 分支（本仓库只跑 Gitea Actions），
  四个 job 的 `runs-on` 直写 `amd64`

### Docs (2026-10-15)
- README：修正 `ClassicLEACH(network)`、`python main.py --config ...`（main.py 不接收任何参数）、
  `results['total_energy']`（是逐轮列表不是标量）、`NetworkAnimator`（实际是
  `OptimizedNetworkAnimator`，参数也不是 `num_frames`）；补齐 5 个示例脚本的入口与说明、
  补全项目结构、新增「质量门禁（CI）」一节；删掉「Excel 导出已集成」等不实描述
- CONTRIBUTING：black/flake8/mypy 命令与 CI 范围、版本、flags 对齐；
  「mypy 类型检查（可选）」改为硬门禁；覆盖率一节区分「CI 强制的 44% 地板」与
  「分模块目标（尚未达到）」；修正示例测试里 `ClassicLEACH(network, cluster_probability=0.05)`
  这类不存在的签名；Python 下限 3.9 → 3.11
- `docs/tutorial.md`：Python 下限、conda 版本、依赖表（补「声明状态」，写清 scipy 未声明、
  seaborn/tqdm/simpy/lightgbm/shap 代码里没有 import）、`--protocol leach-ee` → `leach_ee`
- `config/config.yaml`：注释里把 `leach_ai` 从合法协议名中去掉（注册表里不存在）

### Known Issues (2026-10-15 登记，未修)
- `BatchSimulator.compare_protocols` 的 `summary` 是退化值：`_aggregate_results` 里
  `aggregated["summary"]` 初始化成 `{}` 后从未填充，于是 `lifetimes` 全为 0、
  `best_protocol` 恒等于协议列表的第一个（`examples/advanced_usage.py --demo` 会打印这些 0）。
  属于公开 API 输出变化，需要单独一轮修 + 由 CI 验证
- `config/leach_ai.yaml` 的 `protocol: leach_ai` 未注册进 `LEACHRegistry`，按配置跑会报
  `Unknown protocol`；AI 路径需走 `examples/run_ai_leach.py`
- `requirements.txt` 与 `pyproject.toml` 是两份平行依赖清单：`simpy` 只在 requirements.txt，
  `scipy`（`src/models/network.py` 实际使用）两份都没有；另有无 import 的依赖 seaborn /
  tqdm / simpy / lightgbm / shap
- `src/config/validator.py`（`FullConfig.from_yaml`）没有任何入口调用，覆盖率 0%，
  `cli.py` 用的是自己的 `load_config`
- `COMPLIANCE.md` 声明「禁止用于任何商业目的」，与仓库根 `LICENSE` 的 MIT 授权冲突，
  需要项目所有者决定以哪个为准（未擅自改动法律声明）

## [1.0.0] - 2026-04-16

### Added
- 经典 LEACH 协议实现
- LEACH-C 集中式协议
- LEACH-EE 能量均衡协议
- LEACH-M 移动节点协议
- First Order Radio Model 能量模型
- AI 簇头选择器 (sklearn, PyTorch)
- 事件驱动仿真引擎
- 并行实验运行器
- 动态可视化动画
- 指标图表绘制
- 对比实验工具
- 训练数据生成器
- 不平衡采样器
- CLI 命令行接口
- 配置文件支持
- pytest 单元测试
- 完整文档

### Features
- 多 LEACH 变体支持
- 能量感知路由
- AI 优化簇头选择
- 特征重要性分析
- 可复现仿真
- 配置驱动
