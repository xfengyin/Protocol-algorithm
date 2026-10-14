# Changelog

## [Unreleased] - 2026-10-14

### Fixed
- `models/network.py`: 删除 `self._spatial_index.cKDTree__repr__(positions)`
  —— 该方法并不存在（运行时必 AttributeError），且结果变量从未被使用
- `data/sampler.py`: `undersample()` 补上缺失的 `y_pos = y[pos_mask]`
  —— 原代码在 `np.concatenate([y_pos, y_neg])` 处引用了未定义的名字，一调用就 NameError
  （该行被误放在了 `oversample()`，而那里并不需要它）
- `models/network.py`: `stop_condition` 的注解由内置函数 `callable` 改为 `Callable[["Network", int], bool]`
- `ai/feature_engineering.py`: 邻居缓存类型由 `List[int]` 修正为 `List[Node]`（实际存的是节点列表）
- `simulation/engine.py`: 不再把 `str` 类型的 `output_path` 参数重新赋值为 `Path`

### Changed
- 全量 `black` + `isort` 格式化（30 个文件）
- 删除 46 处未使用 import、4 处赋值后未使用的局部变量、2 处重复 import
- 新增 `.flake8`，`max-line-length = 100` 与 `[tool.black]` 对齐
  （此前 CI 用 88、black 用 100，同一份代码在两套标准下互相打架）
- `requires-python` 由 `>=3.9` 改为 `>=3.11`：`matplotlib>=3.11.2` 要求 Python>=3.11，与原先声明的 3.9 自相矛盾，导致 3.9/3.10 的测试永远卡在安装依赖
- CI 测试矩阵由 `[3.9, 3.10, 3.11]` 改为 `[3.11, 3.12]`
- mypy 增加 `--explicit-package-bases`；存量 47 个类型错误暂以 `continue-on-error` 暴露而不阻塞

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
