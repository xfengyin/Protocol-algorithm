# Changelog

## [Unreleased]

### Added
- `src/benchmark.py`: serial vs thread-pool vs process-pool timing over the same simulation config
  (merged from the former LEACH repository, adapted to this API; no baked-in numbers).
- `tests/test_benchmark_merged.py`: reproducibility across execution modes, result-shape checks, and an
  import-isolation guard with a negative control so it cannot pass vacuously.
- `NOTICE`: provenance of the merged material.

### Fixed
- CI: `security` now `needs: [lint]`. Both jobs started within ~6s and cloned the same cached `setup-python`
  action directory; the loser failed `Set up Python` with `lstat .../.prettierrc.js: no such file`.
  History: 4 of 136 `Set up Python` steps failed, all 4 in `Code Quality Check` (4/36; Security 0/34, Test 0/66).
- Circular import: `import src.leach.classic` (or `src.leach`) as the first import raised
  `ImportError: cannot import name 'ClassicLEACH' from partially initialized module`, because
  `models.network` imported `leach.variants` at module level while `leach.*` imports `models.*`.
  `LEACHRegistry` is now imported lazily inside `Network.__init__`. Found by the new import-isolation
  guard; existing tests never hit it because they all import `src.models.network` first.


（三轮未发布改动：2026-10-14 的代码修复轮 + 2026-10-15 的门禁加固轮 + 2026-10-17 的单一真相源轮）

### Removed (2026-10-17)

- **删除 `src/data/` 整包（319 行：`__init__.py` + `generator.py` + `sampler.py`）**。
  `DataGenerator` / `ImbalancedSampler` 在全仓**零导入点** —— `from src.data` 一条都没有，
  `cli.py`、`main.py`、5 个 examples、8 个 tests 全都不引用，也没有动态导入。
  同时它是 0% 测试覆盖（`CONTRIBUTING.md` 自己的覆盖率表就写着「当前 0%」）。
  真正在用的训练数据生成逻辑是 `examples/run_ai_leach.py::generate_training_data`，
  文档已改为指向它。需要时从 git 历史取回。

- **删除 `src/visualization/animator.py`（478 行）**。`OptimizedNetworkAnimator` /
  `AnimatorConfig` 从任何入口都不可达：`cli.py`、`main.py`、5 个 examples、8 个测试文件
  里都没有调用（只有 `src/visualization/__init__.py` 把它 re-export 成 `NetworkAnimator`，
  而那个别名同样无人使用）。同时它是**导入热路径上的死代码** ——
  `from src.visualization.metrics_plots import MetricsPlotter` 会先执行父包 `__init__.py`，
  于是任何只想画指标图的代码都会连带 import 478 行动画代码和 `matplotlib.animation`。
  README 里那一节还把它写成了 `Animator`（实际类名是 `OptimizedNetworkAnimator`），
  长期没人发现，正好说明这段没有真实使用者。需要时从 git 历史取回。
- **整块删除 `src/config/validator.py`（351 行）**。它是一份**没人调用、且与真实配置
  schema 不兼容**的校验层：认 `energy` / `visualization`，而 `config/*.yaml` 用的是
  `energy_model` / `protocol` / `leach*` / `output`；`validate_config_file` 只被重新导出、
  从无调用方；`FullConfig` 从未被任何入口使用。
  它同时制造了两处同名不同物：`SimulationConfig`（真的那个在 `src/simulation/engine.py`）
  与 `model_type`（这里指 `sklearn`/`pytorch`，`src/ai/sklearn_selector.py` 里指 `rf`/`gb`）。
  唯一真实被使用的是 `SUPPORTED_MODEL_TYPES`，已随它唯一的使用者移到 `src/ai/trainer.py`。
- 一并修正的**文档与现实不符**：`CONTRIBUTING.md` 与 `docs/api_reference.md` 都写了
  `from src.config.validator import ConfigValidator` —— 这个类**从未存在过**；
  `docs/api_reference.md` §9 改为如实描述真实入口 `cli.load_config`
  （含「不做 schema 校验」这一现状）。

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

#### 依赖审计门禁第一次真实运行就拦下了两条漏洞
- run 10220 / task 1482：`pytest 8.1.1` → **PYSEC-2026-1845**（修复版本 9.0.3，8.x 无修复版）、
  `setuptools 79.0.1` → **PYSEC-2026-3447**（修复版本 83.0.0）
- 处置是**升版本**而不是加豁免：测试工具 pin 改为 `pytest==9.1.1 pytest-cov==7.1.0
  pytest-xdist==3.8.0`，安装阶段改为 `--upgrade pip setuptools`；
  豁免（`--ignore-vuln`）只在升版本不可行时使用，且必须写明「为什么留 + 何时复查」
- 这条门禁的已知盲区也写进了 workflow 注释：带本地版本号的包会被 pip-audit 跳过
  （实测 `torch 2.14.1+cpu` 报 "Dependency not found on PyPI and could not be audited"），
  即 torch 的漏洞不在覆盖内 —— 需要时用 `pip-audit --disable-pip --no-deps -r requirements.txt` 补审

#### 依赖瘦身与 CI 成本（共享 runner 上的实际收益）
- `requirements.txt` 删掉 **7 个零 import 的依赖**：`jupyter` / `ipykernel` / `tqdm` / `simpy` /
  `shap` / `seaborn` / `lightgbm`。判据是 `git ls-files '*.py' | xargs grep -nE '^\s*(import|from) <包名>'`
  零命中，并逐一核过字符串误报：`shap` 的命中全是 `.shape[..]`，`seaborn` 只有
  `style="seaborn-v0_8-darkgrid"`（matplotlib 自带样式名），`lightgbm` 只有一处校验字符串。
  光是 `jupyter` 一项就带进约 80 个传递依赖 —— test job 的安装阶段长期在 20 分钟量级，
  而 NAS 上只有 2 个 amd64 槽位，一个仓的 run 会卡住其他仓的验收
- `torch` 改由 CI 显式安装 **CPU 轮子**：PyPI 上 Linux 的默认轮子会额外拉 `nvidia-*` CUDA
  依赖（约 2.5GB），而 runner 是纯 CPU 机器。实测 CPU 轮子 196MB、METADATA 里 nvidia 依赖
  条数为 0，`download.pytorch.org` 从 NAS 直连可达（HTTP 200 / 3.3s）
- 补上 `scipy` 的显式声明（`src/models/network.py` 的 `cKDTree` 真实使用，此前只靠
  scikit-learn 传递带入；scikit-learn 自身依赖 scipy，所以不增加任何安装量）
- `pyproject.toml` 的可选依赖同步收窄：删掉 `simulation` extra（唯一成员是没被 import 的
  `simpy`）、`ai` extra 只留 `torch`、`dev` 去掉 `jupyter`/`ipykernel`
- 矩阵 `3.11 + 3.12` **保留**，并把依据写进 workflow 注释：`requires-python = ">=3.11"` 是
  无上界的对外声明，两条腿分别是「声明窗口下界/上界各一份证据」，砍掉 3.12 等于声明了却没证据；
  成本大头是安装而非测试，所以省的是安装。判据：两腿合计仍 >15 分钟才考虑砍腿
- `model_type` 的合法值从三份（两份互相矛盾）收敛为**一处**：
  `src/config/validator.py::SUPPORTED_MODEL_TYPES = ("sklearn", "pytorch")` 由
  `src/ai/trainer.py` 消费；`lightgbm` 从合法值中删除（它从未被实现，此前用户会
  「校验通过 → 构造时抛 Unknown model type」，这条死分支也是 `lightgbm` 依赖的去处）

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

### Fixed (2026-10-15 第二轮)
- **`BatchSimulator.compare_protocols` 的 `summary` 不再返回退化值**：`_aggregate_results` 里
  `aggregated["summary"]` 此前初始化成 `{}` 后从未填充，于是 `lifetimes` 全为 0、
  `best_protocol` 恒等于协议列表的第一个（`max()` 在相同 key 上返回第一个元素）。
  这个错值会被 `examples/advanced_usage.py`（README 示例脚本表里列着）打印、
  也被 `docs/tutorial.md` 演示，属于「公开输出恒错」，故本轮修掉：
  `summary` 填 `network_lifetime` / `half_network_lifetime` / `avg_energy_consumption`
  （各自是全部运行的均值，保持标量 —— `compare_protocols` 要把它们放进 `max(..., key=...)`）。
  回归测试见 `tests/test_batch_simulator.py`（断言较优协议排在列表第二个时会被选中）
- **打包布局修复**：`[tool.setuptools.packages.find]` 从 `where = ["src"]` 改为
  `where = ["."], include = ["src", "src.*"]`，并用 `py-modules` 显式打进 `main.py` / `cli.py`。
  此前 wheel 的顶层包是 `ai/ models/ config/ ...`（src/ 里的目录被平铺），而全仓 45 处写的是
  `from src.xxx import ...` ⇒ `pip install .` 之后 `import src.models` 必然 `ModuleNotFoundError`；
  没被发现是因为从仓库根目录运行时 CWD 在 `sys.path` 里、恰好命中仓库里的 `src/`。
  CI 新增「Verify built wheel is importable (packaging)」一步：构建 wheel → 装进 venv →
  **在非仓库目录**下 `import src, src.models, src.ai, src.simulation`，并断言导入到的不是仓库里的那份

### Known Issues (2026-10-15 登记，未修)
- `config/leach_ai.yaml` 的 `protocol: leach_ai` 未注册进 `LEACHRegistry`，按配置跑会报
  `Unknown protocol`；AI 路径需走 `examples/run_ai_leach.py`
- `requirements.txt` 与 `pyproject.toml` 仍是两份清单（内容已对齐：pyproject 的 dependencies
  是 requirements 的子集 + 测试工具），理想是合成一份或由 `-e .` 生成，需要所有者定口径
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
