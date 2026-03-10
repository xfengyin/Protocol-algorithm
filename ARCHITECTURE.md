# Protocol-algorithm v2.0 重构架构设计

## 🎯 项目定位

**无线传感器网络 (WSN) 协议算法仿真平台** - 现代化、高性能、可视化

---

## 🏗️ 架构概览

```
Protocol-algorithm v2.0
├── core/                    # Rust 核心算法层
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs          # 库入口
│   │   ├── leach/          # LEACH 协议实现
│   │   │   ├── mod.rs
│   │   │   ├── node.rs     # 节点定义
│   │   │   ├── cluster.rs  # 簇头管理
│   │   │   ├── energy.rs   # 能量模型
│   │   │   └── protocol.rs # 协议逻辑
│   │   ├── network/        # 网络拓扑
│   │   │   ├── mod.rs
│   │   │   ├── topology.rs # 拓扑生成
│   │   │   └── grid.rs     # 网格布局
│   │   ├── simulation/     # 仿真引擎
│   │   │   ├── mod.rs
│   │   │   ├── engine.rs   # 仿真循环
│   │   │   └── metrics.rs  # 指标统计
│   │   └── utils/          # 工具函数
│   │       ├── mod.rs
│   │       ├── random.rs   # 随机数生成
│   │       └── math.rs     # 数学计算
│   └── tests/              # Rust 单元测试
│
├── python/                  # Python 绑定层 (PyO3)
│   ├── pyproject.toml
│   ├── src/
│   │   ├── lib.rs          # PyO3 绑定
│   │   └── wrapper.rs      # Python API 封装
│   ├── protocol_algo/      # Python 包
│   │   ├── __init__.py
│   │   ├── leach.py        # LEACH Python API
│   │   ├── network.py      # 网络 API
│   │   └── viz.py          # 可视化 API
│   └── examples/           # Python 示例
│       ├── basic_leach.py
│       ├── ai_assisted.py
│       └── custom_topology.py
│
├── web/                     # Web 前端层
│   ├── Cargo.toml          # Axum 后端
│   ├── src/
│   │   ├── main.rs         # Web 服务器入口
│   │   ├── api/            # REST API
│   │   │   ├── mod.rs
│   │   │   ├── simulation.rs
│   │   │   └── visualization.rs
│   │   └── ws/             # WebSocket
│   │       └── mod.rs      # 实时推送
│   ├── frontend/           # React 前端
│   │   ├── package.json
│   │   ├── src/
│   │   │   ├── App.tsx
│   │   │   ├── components/
│   │   │   │   ├── NetworkViz.tsx    # 网络可视化
│   │   │   │   ├── ClusterView.tsx   # 簇头视图
│   │   │   │   ├── MetricsChart.tsx  # 指标图表
│   │   │   │   └── ControlPanel.tsx  # 控制面板
│   │   │   ├── hooks/
│   │   │   │   └── useSimulation.ts
│   │   │   └── styles/
│   │   │       └── index.css
│   │   └── public/
│   │       └── index.html
│   └── dist/               # 构建输出
│
├── viz/                     # Python 可视化脚本
│   ├── requirements.txt
│   ├── static_viz.py       # 静态图表 (Matplotlib)
│   ├── interactive_viz.py  # 交互图表 (Plotly)
│   └── animations.py       # 动画生成
│
├── docs/                    # 文档
│   ├── README.md
│   ├── ARCHITECTURE.md
│   ├── API.md
│   └── examples/
│
├── scripts/                 # 工具脚本
│   ├── build.sh
│   ├── test.sh
│   └── demo.sh
│
├── Cargo.toml              # Rust Workspace
├── pyproject.toml          # Python 项目配置
└── README.md               # 项目说明
```

---

## 🎨 可视化设计

### 1. 网络拓扑图 (D3.js / Plotly)

**风格：** 现代、简洁、科技感

**配色方案：**
```
主色调：  #2563EB (蓝色 - 普通节点)
簇头色：  #DC2626 (红色 - 簇头节点)
基站色：  #16A34A (绿色 - Base Station)
连接线：  #94A3B8 (灰色 - 通信链路)
背景：    #F8FAFC (浅灰 - 画布背景)
```

**视觉元素：**
- 节点：圆形，带光晕效果
- 簇头：脉冲动画
- 通信链路：渐变线条
- 数据流：粒子动画

### 2. 指标仪表盘 (React + Recharts)

**展示指标：**
- 网络存活时间
- 能量消耗曲线
- 簇头轮换统计
- 数据包成功率

### 3. 3D 地形视图 (Three.js - 可选)

- 节点高度映射
- 能量热力图
- 信号强度可视化

---

## 🔧 技术栈详解

### 核心层 (Rust)

| 组件 | 库 | 用途 |
|------|-----|------|
| 数值计算 | `ndarray` | 矩阵运算 |
| 随机数 | `rand` | 随机分布 |
| 序列化 | `serde` + `serde_json` | 数据交换 |
| 并行计算 | `rayon` | 多线程仿真 |
| Python 绑定 | `pyo3` | Python 接口 |

### Web 后端 (Rust)

| 组件 | 库 | 用途 |
|------|-----|------|
| Web 框架 | `axum` | HTTP 服务 |
| WebSocket | `tokio-tungstenite` | 实时推送 |
| 静态文件 | `tower-http` | 前端服务 |
| 异步运行时 | `tokio` | 异步 IO |

### Web 前端 (TypeScript)

| 组件 | 库 | 用途 |
|------|-----|------|
| 框架 | `React 18` | UI 组件 |
| 可视化 | `D3.js v7` | 网络拓扑图 |
| 图表 | `Recharts` | 指标图表 |
| 3D | `Three.js` | 3D 视图 |
| 样式 | `Tailwind CSS` | 快速样式 |
| 状态 | `Zustand` | 状态管理 |

### Python 层

| 组件 | 库 | 用途 |
|------|-----|------|
| 绑定 | `PyO3` | Rust 调用 |
| 静态图 | `Matplotlib` | 论文图表 |
| 交互图 | `Plotly` | 网页图表 |
| 动画 | `Manim` | 教学动画 |

---

## 📊 API 设计

### REST API

```
GET  /api/simulations          # 获取仿真列表
POST /api/simulations          # 创建仿真
GET  /api/simulations/:id      # 获取仿真详情
POST /api/simulations/:id/run  # 运行仿真
GET  /api/simulations/:id/results  # 获取结果

GET  /api/visualizations/network   # 网络拓扑图数据
GET  /api/visualizations/metrics   # 指标数据
GET  /api/visualizations/energy    # 能量热力图
```

### WebSocket

```
/ws/simulation/:id  # 实时推送仿真进度
```

### Python API

```python
from protocol_algo import LEACH, Network, Visualizer

# 创建网络
network = Network(nodes=100, area=100, base_station=(50, 150))

# 配置 LEACH
leach = LEACH(
    p=0.05,           # 簇头概率
    rounds=100,       # 仿真轮数
    energy_model='first_order'
)

# 运行仿真
results = leach.run(network)

# 可视化
viz = Visualizer(style='modern')
viz.plot_network(network, clusters=results.clusters)
viz.plot_metrics(results)
viz.save('output.png')
```

---

## 🎯 重构阶段

### Phase 1: Rust 核心 (2 周)
- [ ] 项目骨架搭建
- [ ] LEACH 核心算法实现
- [ ] 单元测试
- [ ] 性能基准测试

### Phase 2: Python 绑定 (1 周)
- [ ] PyO3 绑定
- [ ] Python API 封装
- [ ] 示例脚本
- [ ] 文档

### Phase 3: Web 后端 (1 周)
- [ ] Axum 服务器
- [ ] REST API
- [ ] WebSocket
- [ ] 数据序列化

### Phase 4: Web 前端 (2 周)
- [ ] React 项目搭建
- [ ] D3.js 网络可视化
- [ ] 指标仪表盘
- [ ] 控制面板
- [ ] 样式优化

### Phase 5: Python 可视化 (1 周)
- [ ] Matplotlib 静态图
- [ ] Plotly 交互图
- [ ] 动画生成

### Phase 6: 文档与发布 (1 周)
- [ ] README 编写
- [ ] API 文档
- [ ] 示例教程
- [ ] GitHub Release

**总计：8 周**

---

## 📈 性能目标

| 指标 | v1.0 | v2.0 目标 |
|------|------|----------|
| 仿真速度 (1000 节点) | ~5s | <0.5s |
| 内存占用 | ~200MB | <50MB |
| Web 渲染 FPS | N/A | 60 FPS |
| Python 调用延迟 | N/A | <1ms |

---

## 🎨 UI/UX 设计原则

1. **简洁直观** - 一键运行，实时反馈
2. **数据驱动** - 所有可视化基于真实数据
3. **响应式** - 适配桌面/平板/手机
4. **暗色模式** - 护眼，专业感
5. **动画流畅** - 60 FPS，无卡顿

---

## 📦 交付物

### 代码
- ✅ Rust 核心库
- ✅ Python 包 (PyPI 可发布)
- ✅ Web 应用 (可部署)
- ✅ 完整测试套件

### 文档
- ✅ README (中英双语)
- ✅ API 文档
- ✅ 架构文档
- ✅ 示例教程

### 可视化
- ✅ 网络拓扑图 (D3.js)
- ✅ 指标仪表盘 (Recharts)
- ✅ 静态图表 (Matplotlib)
- ✅ 交互图表 (Plotly)
- ✅ 演示动画 (可选)

### 部署
- ✅ Docker 镜像
- ✅ GitHub Actions CI/CD
- ✅ GitHub Pages 演示

---

## 🔗 参考项目

- [LEACH Protocol](https://en.wikipedia.org/wiki/Low_Energy_Adaptive_Clustering_Hierarchy)
- [D3.js Gallery](https://observablehq.com/@d3/gallery)
- [PyO3 Guide](https://pyo3.rs/)
- [Axum Examples](https://github.com/tokio-rs/axum/tree/main/examples)

---

_Protocol-algorithm v2.0 - 让协议仿真更优雅_
