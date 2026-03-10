# Protocol-algorithm v2.0

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI/CD](https://github.com/xfengyin/Protocol-algorithm/actions/workflows/ci.yml/badge.svg)](https://github.com/xfengyin/Protocol-algorithm/actions)

**无线传感器网络 (WSN) 协议算法仿真平台** - 高性能、现代化可视化

---

## ✨ 特性

- 🚀 **Rust 核心** - 极致性能，1000 节点仿真 <0.5s
- 🐍 **Python 绑定** - 科研友好，简单易用
- 🌐 **Web 可视化** - React + D3.js，60 FPS 流畅动画
- 📊 **交互图表** - Plotly 交互式图表，美观专业
- 🎨 **现代审美** - 科技感设计，暗色模式
- 🐳 **Docker 支持** - 一键部署，开箱即用

---

## 📦 快速开始

### 方式 1: Docker (推荐)

```bash
# 克隆仓库
git clone https://github.com/xfengyin/Protocol-algorithm.git
cd Protocol-algorithm

# 启动所有服务
docker-compose up -d

# 访问 Web 界面
open http://localhost:3000

# 访问 Jupyter Notebook
open http://localhost:8888
```

### 方式 2: Python 包

```bash
# 安装 Rust (必需)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 安装 Python 包
pip install protocol-algo

# 运行示例
python -c "from protocol_algo import Network, LEACH; n=Network(); l=LEACH(); r=l.run(n); print(f'存活率：{r.survival_rate():.1f}%')"
```

### 方式 3: 源码构建

```bash
# 克隆仓库
git clone https://github.com/xfengyin/Protocol-algorithm.git
cd Protocol-algorithm

# 一键构建
./scripts/build.sh all

# 运行 Web 界面
cd web && cargo run --release
# 访问 http://localhost:3000
```

---

## 🚀 使用示例

### Python API

```python
from protocol_algo import Network, LEACH, Visualizer

# 创建网络
network = Network(nodes=100, area=100, base_station=(50, 150))

# 配置 LEACH
leach = LEACH(p=0.05, rounds=100, initial_energy=0.5)

# 运行仿真
result = leach.run(network)
print(f"初始节点：{result.initial_nodes}")
print(f"最终存活：{result.final_alive}")
print(f"存活率：{result.survival_rate():.1f}%")

# 可视化
viz = Visualizer(style="modern")
viz.plot_network(network, result)
viz.plot_metrics(result)
viz.save("output.png")
```

### Rust API

```rust
use protocol_algo_core::{LEACH, LEACHConfig, Simulation, Topology};

let config = LEACHConfig {
    p: 0.05,
    rounds: 100,
    initial_energy: 0.5,
    base_station: (50.0, 150.0),
    seed: 42,
};

let topology = Topology::random(100.0, 100.0, config.base_station);
let mut simulation = Simulation::new(config, topology, 100);
let states = simulation.run();

println!("最终存活节点：{}", states.last().unwrap().alive_nodes);
```

### Web 界面

访问 `http://localhost:3000` 打开交互式 Web 界面

- 调节节点数量、仿真轮数、簇头概率
- 实时查看网络拓扑图
- 查看能量消耗曲线
- 导出仿真结果

---

## 📊 可视化展示

### 网络拓扑图

- **节点**: 蓝色圆形，带光晕效果
- **簇头**: 红色，脉冲动画
- **基站**: 绿色三角形
- **链路**: 渐变贝塞尔曲线

### 指标仪表盘

- 网络存活时间曲线
- 能量消耗统计
- 簇头轮换动画

---

## 🏗️ 架构

```
Protocol-algorithm v2.0
├── core/ (Rust)          # 核心算法
├── python/ (PyO3)        # Python 绑定
├── web/ (TypeScript)     # Web 前端
├── viz/ (Python)         # 可视化脚本
├── docker-compose.yml    # Docker 编排
└── docs/                 # 文档
```

---

## 📈 性能对比

| 指标 | v1.0 | v2.0 | 提升 |
|------|------|------|------|
| 100 节点仿真 | ~0.5s | <0.05s | **10x** |
| 1000 节点仿真 | ~5s | <0.5s | **10x** |
| 内存占用 | ~200MB | <50MB | **75%** |
| Web 渲染 FPS | N/A | 60 | **流畅** |

---

## 🎨 配色方案

```
节点：    #2563EB (蓝色)
簇头：    #DC2626 (红色)
基站：    #16A34A (绿色)
链路：    #94A3B8 (灰色)
背景：    #F8FAFC (浅灰)
文字：    #1E293B (深灰)
```

---

## 📖 文档

- [架构设计](docs/ARCHITECTURE.md)
- [开发指南](docs/DEVELOPER-GUIDE.md)
- [API 参考](docs/API.md)
- [示例教程](python/examples/)

---

## 🧪 测试

```bash
# Rust 测试
cd core && cargo test

# Python 测试
cd python && pytest

# Web 测试
cd web/frontend && npm test

# 可视化测试
cd viz && python demo_viz.py
```

---

## 🐳 Docker

```bash
# 构建镜像
docker build -t protocol-algo:2.0 .

# 运行容器
docker run -p 3000:3000 protocol-algo:2.0

# 使用 Docker Compose
docker-compose up -d
```

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

1. Fork 仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---

## 🔗 链接

- GitHub: https://github.com/xfengyin/Protocol-algorithm
- PyPI: https://pypi.org/project/protocol-algo/
- 文档：https://xfengyin.github.io/Protocol-algorithm/

---

## 🙏 致谢

感谢使用 Protocol-algorithm v2.0！

---

_Protocol-algorithm v2.0 - 让协议仿真更优雅_ ✨
