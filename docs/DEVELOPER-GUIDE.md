# Protocol-algorithm v2.0 开发指南

## 🚀 快速开始

### 环境要求

- **Rust**: 1.75+
- **Python**: 3.10+
- **Node.js**: 18+
- **构建工具**: make, cmake (可选)

### 一键构建

```bash
# 克隆仓库
git clone https://github.com/xfengyin/Protocol-algorithm.git
cd Protocol-algorithm

# 运行构建脚本
./scripts/build.sh all
```

### 分步构建

#### 1. 构建 Rust 核心

```bash
cd core

# 开发构建
cargo build

# 发布构建
cargo build --release

# 运行测试
cargo test

# 运行基准测试
cargo bench
```

#### 2. 构建 Python 绑定

```bash
cd python

# 安装 maturin
pip install maturin

# 开发模式安装
maturin develop

# 构建 wheel
maturin build --release

# 发布到 PyPI
maturin publish
```

#### 3. 构建 Web 前端

```bash
cd web/frontend

# 安装依赖
npm install

# 开发服务器
npm run dev

# 生产构建
npm run build

# 代码检查
npm run lint
```

#### 4. 运行可视化演示

```bash
cd viz

# 安装依赖
pip install -r requirements.txt

# 运行演示
python demo_viz.py
```

---

## 📖 使用示例

### Python API

```python
from protocol_algo import Network, LEACH, Visualizer

# 创建网络
network = Network(nodes=100, area=100.0)

# 配置 LEACH
leach = LEACH(p=0.05, rounds=100)

# 运行仿真
result = leach.run(network)
print(f"存活率：{result.survival_rate():.1f}%")

# 可视化
viz = Visualizer(style="modern")
viz.plot_network(network, result)
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

```bash
# 启动后端
cd web
cargo run --release

# 启动前端 (新终端)
cd web/frontend
npm run dev

# 访问 http://localhost:5173
```

---

## 🧪 测试

### Rust 测试

```bash
cd core

# 单元测试
cargo test

# 集成测试
cargo test --test integration_tests

# 测试覆盖率 (需要 cargo-tarpaulin)
cargo tarpaulin --out Html
```

### Python 测试

```bash
cd python

# 安装测试依赖
pip install pytest pytest-benchmark

# 运行测试
pytest

# 基准测试
pytest --benchmark-only
```

### Web 测试

```bash
cd web/frontend

# 单元测试
npm test

# E2E 测试 (需要 Playwright)
npx playwright test
```

---

## 📊 性能优化

### Rust 优化

```toml
# Cargo.toml
[profile.release]
lto = true
codegen-units = 1
opt-level = 3
```

### Python 优化

- 使用 NumPy 向量化操作
- 避免 Python 循环，使用 C 扩展
- 使用 PyO3 调用 Rust 核心

### Web 优化

- 使用 Web Workers 进行后台计算
- 使用 Canvas 而非 SVG 渲染大量节点
- 启用 gzip/brotli 压缩

---

## 🤝 贡献指南

### 提交代码

1. Fork 仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 代码规范

**Rust:**
```bash
cargo fmt
cargo clippy
```

**Python:**
```bash
black .
ruff check .
```

**TypeScript:**
```bash
npm run lint
```

---

## 📚 架构说明

详见 [ARCHITECTURE.md](./ARCHITECTURE.md)

---

## 🔗 相关链接

- GitHub: https://github.com/xfengyin/Protocol-algorithm
- PyPI: https://pypi.org/project/protocol-algo/
- 文档：https://xfengyin.github.io/Protocol-algorithm/

---

_Protocol-algorithm v2.0 - 让协议仿真更优雅_
