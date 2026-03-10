# Protocol-algorithm v2.0 重构计划

## 🚀 项目启动

**任务：** Protocol-algorithm 项目重构 - 方案 D（混合架构）

**目标：**
- ✅ Rust 核心算法（高性能）
- ✅ Python 绑定（科研友好）
- ✅ Web 前端（React + D3.js 最美可视化）
- ✅ Python 可视化（Matplotlib/Plotly）

**核心要求：** **模拟结果图片美观漂亮，符合现代审美标准**

---

## 📋 执行清单

### Phase 1: Rust 核心 ⏳ 进行中
- [ ] 创建 Rust workspace
- [ ] 实现 LEACH 核心算法
  - [ ] 节点定义 (Node)
  - [ ] 能量模型 (Energy Model)
  - [ ] 簇头选择 (Cluster Head Selection)
  - [ ] 协议逻辑 (Protocol Logic)
- [ ] 网络拓扑生成
  - [ ] 随机分布
  - [ ] 网格分布
  - [ ] 自定义分布
- [ ] 仿真引擎
  - [ ] 轮次循环
  - [ ] 指标统计
- [ ] 单元测试

### Phase 2: Python 绑定
- [ ] PyO3 绑定配置
- [ ] Python API 封装
- [ ] 示例脚本
- [ ] PyPI 发布准备

### Phase 3: Web 后端
- [ ] Axum 服务器搭建
- [ ] REST API 实现
- [ ] WebSocket 实时推送
- [ ] 数据序列化

### Phase 4: Web 前端 🎨 重点
- [ ] React 项目初始化
- [ ] D3.js 网络拓扑可视化
  - [ ] 节点渲染（带光晕）
  - [ ] 簇头脉冲动画
  - [ ] 通信链路渐变
  - [ ] 数据流粒子动画
- [ ] 指标仪表盘 (Recharts)
- [ ] 控制面板
- [ ] Tailwind CSS 样式
- [ ] 暗色模式

### Phase 5: Python 可视化
- [ ] Matplotlib 静态图表
  - [ ] 网络拓扑图
  - [ ] 能量消耗曲线
  - [ ] 簇头轮换统计
- [ ] Plotly 交互图表
  - [ ] 3D 地形视图
  - [ ] 能量热力图
- [ ] 动画生成 (Manim 可选)

### Phase 6: 文档与发布
- [ ] README (中英双语)
- [ ] API 文档
- [ ] 示例教程
- [ ] GitHub Release
- [ ] GitHub Pages 演示

---

## 🎨 视觉设计规范

### 配色方案
```
主色调 (节点):    #2563EB (蓝色)
簇头色：          #DC2626 (红色)
基站色：          #16A34A (绿色)
连接线：          #94A3B8 (灰色)
背景：            #F8FAFC (浅灰)
文字：            #1E293B (深灰)
强调色：          #8B5CF6 (紫色)
```

### 视觉元素
- **节点**: 圆形，半径 8px，带 2px 光晕
- **簇头**: 红色，脉冲动画 (1s 周期)
- **基站**: 绿色三角形，带标签
- **链路**: 贝塞尔曲线，渐变透明度
- **数据流**: 粒子动画，沿链路移动

### 字体
- 英文：Inter / SF Pro Display
- 中文：思源黑体 / 霞鹜文楷

---

## 📊 性能目标

| 场景 | v1.0 | v2.0 目标 |
|------|------|----------|
| 100 节点仿真 | ~0.5s | <0.05s |
| 1000 节点仿真 | ~5s | <0.5s |
| Web 渲染 FPS | N/A | 60 FPS |
| 内存占用 | ~200MB | <50MB |

---

## 🔗 参考资源

- [D3.js Gallery](https://observablehq.com/@d3/gallery)
- [Plotly Python](https://plotly.com/python/)
- [PyO3 Guide](https://pyo3.rs/)
- [Axum Examples](https://github.com/tokio-rs/axum/tree/main/examples)

---

_开始执行：Phase 1 - Rust 核心_
