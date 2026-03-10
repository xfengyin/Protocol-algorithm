#!/usr/bin/env python3
"""
Protocol-algorithm v2.0 - 现代化可视化演示

生成美观的 LEACH 协议仿真结果图
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import PatchCollection
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Tuple, Dict

# 配色方案 - 现代科技感
COLORS = {
    'node': '#2563EB',        # 蓝色 - 普通节点
    'cluster_head': '#DC2626', # 红色 - 簇头
    'base_station': '#16A34A', # 绿色 - 基站
    'link': '#94A3B8',        # 灰色 - 链路
    'bg': '#F8FAFC',          # 浅灰 - 背景
    'text': '#1E293B',        # 深灰 - 文字
    'accent': '#8B5CF6',      # 紫色 - 强调
}

def generate_node_positions(n_nodes: int, area_size: float = 100.0, seed: int = 42) -> np.ndarray:
    """生成随机节点位置"""
    np.random.seed(seed)
    x = np.random.uniform(0, area_size, n_nodes)
    y = np.random.uniform(0, area_size, n_nodes)
    return np.column_stack([x, y])

def select_cluster_heads(positions: np.ndarray, p: float = 0.05, seed: int = 42) -> np.ndarray:
    """选择簇头"""
    np.random.seed(seed + 1)
    n_nodes = len(positions)
    is_ch = np.random.random(n_nodes) < p
    # 确保至少有一个簇头
    if not np.any(is_ch):
        is_ch[np.random.randint(n_nodes)] = True
    return is_ch

def plot_network_matplotlib(positions: np.ndarray, 
                            cluster_heads: np.ndarray,
                            base_station: Tuple[float, float] = (50, 150),
                            save_path: str = 'network_viz.png',
                            dpi: int = 300):
    """
    使用 Matplotlib 绘制网络拓扑图
    风格：现代、简洁、科技感
    """
    fig, ax = plt.subplots(figsize=(12, 10), facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])
    
    # 绘制通信链路 (节点到簇头)
    ch_positions = positions[cluster_heads]
    for i, pos in enumerate(positions):
        if not cluster_heads[i]:
            # 找到最近的簇头
            distances = np.linalg.norm(ch_positions - pos, axis=1)
            nearest_ch_idx = np.argmin(distances)
            nearest_ch = ch_positions[nearest_ch_idx]
            
            ax.plot([pos[0], nearest_ch[0]], 
                   [pos[1], nearest_ch[1]], 
                   '-', 
                   color=COLORS['link'], 
                   linewidth=0.5, 
                   alpha=0.3,
                   zorder=1)
    
    # 绘制普通节点 (带光晕效果)
    for pos in positions[~cluster_heads]:
        # 光晕
        glow = Circle(pos, radius=3, color=COLORS['node'], alpha=0.2)
        ax.add_patch(glow)
        # 节点
        node = Circle(pos, radius=1.5, color=COLORS['node'], alpha=0.8)
        ax.add_patch(node)
    
    # 绘制簇头 (带脉冲效果)
    for pos in ch_positions:
        # 外圈脉冲
        pulse1 = Circle(pos, radius=4, color=COLORS['cluster_head'], alpha=0.3)
        pulse2 = Circle(pos, radius=3, color=COLORS['cluster_head'], alpha=0.5)
        ax.add_patch(pulse1)
        ax.add_patch(pulse2)
        # 簇头
        ch = Circle(pos, radius=2, color=COLORS['cluster_head'], alpha=1.0)
        ax.add_patch(ch)
    
    # 绘制基站
    bs_triangle = Polygon([
        [base_station[0] - 3, base_station[1] + 3],
        [base_station[0] + 3, base_station[1] + 3],
        [base_station[0], base_station[1] - 3]
    ], closed=True, color=COLORS['base_station'], alpha=1.0)
    ax.add_patch(bs_triangle)
    ax.text(base_station[0], base_station[1] - 5, 'BS', 
            ha='center', va='top', fontsize=10, color=COLORS['text'], fontweight='bold')
    
    # 设置
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_aspect('equal')
    ax.set_xlabel('X Position (m)', fontsize=12, color=COLORS['text'])
    ax.set_ylabel('Y Position (m)', fontsize=12, color=COLORS['text'])
    ax.set_title('LEACH Protocol - Network Topology\n(100 nodes, 5% cluster heads)', 
                fontsize=14, color=COLORS['text'], fontweight='bold', pad=20)
    
    # 移除上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['link'])
    ax.spines['bottom'].set_color(COLORS['link'])
    ax.tick_params(colors=COLORS['text'])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, facecolor=COLORS['bg'], bbox_inches='tight')
    plt.close()
    print(f"✅ 网络拓扑图已保存至：{save_path}")

def plot_energy_curve(rounds: List[int], 
                      alive_nodes: List[int],
                      save_path: str = 'energy_curve.png'):
    """
    绘制能量消耗曲线
    """
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])
    
    # 计算存活率
    survival_rate = [n / alive_nodes[0] * 100 for n in alive_nodes]
    
    ax.plot(rounds, survival_rate, '-', 
            color=COLORS['node'], linewidth=2.5, 
            label='Survival Rate', marker='o', markersize=3)
    
    # 标记关键点
    half_life = next((i for i, s in enumerate(survival_rate) if s < 50), None)
    if half_life:
        ax.axvline(x=rounds[half_life], color=COLORS['cluster_head'], 
                  linestyle='--', linewidth=2, alpha=0.5)
        ax.text(rounds[half_life], 55, f'Half Life\nRound {rounds[half_life]}',
               ha='center', color=COLORS['cluster_head'], fontweight='bold')
    
    ax.set_xlabel('Round', fontsize=12, color=COLORS['text'])
    ax.set_ylabel('Survival Rate (%)', fontsize=12, color=COLORS['text'])
    ax.set_title('Network Lifetime - Node Survival Over Time',
                fontsize=14, color=COLORS['text'], fontweight='bold', pad=20)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['link'])
    ax.spines['bottom'].set_color(COLORS['link'])
    ax.tick_params(colors=COLORS['text'])
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor=COLORS['bg'], bbox_inches='tight')
    plt.close()
    print(f"✅ 能量曲线图已保存至：{save_path}")

def plot_network_plotly(positions: np.ndarray,
                        cluster_heads: np.ndarray,
                        base_station: Tuple[float, float] = (50, 150),
                        save_path: str = 'network_interactive.html'):
    """
    使用 Plotly 绘制交互式网络图
    """
    # 普通节点
    normal_nodes = positions[~cluster_heads]
    ch_nodes = positions[cluster_heads]
    
    # 创建链路
    link_x, link_y = [], []
    for pos in normal_nodes:
        distances = np.linalg.norm(ch_nodes - pos, axis=1)
        nearest_ch = ch_nodes[np.argmin(distances)]
        link_x.extend([pos[0], nearest_ch[0], None])
        link_y.extend([pos[1], nearest_ch[1], None])
    
    fig = go.Figure()
    
    # 添加链路
    fig.add_trace(go.Scatter(
        x=link_x, y=link_y,
        mode='lines',
        line=dict(color=COLORS['link'], width=1),
        hoverinfo='skip',
        name='Links'
    ))
    
    # 添加普通节点
    fig.add_trace(go.Scatter(
        x=normal_nodes[:, 0], y=normal_nodes[:, 1],
        mode='markers',
        marker=dict(
            size=10,
            color=COLORS['node'],
            opacity=0.8,
            line=dict(color='white', width=1)
        ),
        hovertemplate='Node: %{customdata}<br>Position: (%{x}, %{y})<extra></extra>',
        customdata=np.arange(len(normal_nodes)),
        name='Normal Nodes'
    ))
    
    # 添加簇头
    fig.add_trace(go.Scatter(
        x=ch_nodes[:, 0], y=ch_nodes[:, 1],
        mode='markers',
        marker=dict(
            size=15,
            color=COLORS['cluster_head'],
            opacity=1.0,
            line=dict(color='white', width=2),
            symbol='star'
        ),
        hovertemplate='<b>Cluster Head</b><br>Position: (%{x}, %{y})<extra></extra>',
        name='Cluster Heads'
    ))
    
    # 添加基站
    fig.add_trace(go.Scatter(
        x=[base_station[0]], y=[base_station[1]],
        mode='markers+text',
        marker=dict(
            size=20,
            color=COLORS['base_station'],
            symbol='triangle-up'
        ),
        text=['<b>BS</b>'],
        textposition='bottom center',
        hovertemplate='<b>Base Station</b><extra></extra>',
        name='Base Station'
    ))
    
    # 布局
    fig.update_layout(
        title=dict(
            text='LEACH Protocol - Interactive Network Visualization',
            font=dict(size=18, color=COLORS['text'])
        ),
        xaxis=dict(
            title='X Position (m)',
            showgrid=True,
            gridcolor=COLORS['link'],
            gridwidth=0.5,
            range=[-5, 105]
        ),
        yaxis=dict(
            title='Y Position (m)',
            showgrid=True,
            gridcolor=COLORS['link'],
            gridwidth=0.5,
            scaleanchor='x',
            scaleratio=1,
            range=[-5, 105]
        ),
        plot_bgcolor=COLORS['bg'],
        paper_bgcolor=COLORS['bg'],
        font=dict(color=COLORS['text']),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor=COLORS['link']
        ),
        height=700,
        width=900
    )
    
    fig.write_html(save_path)
    print(f"✅ 交互式图表已保存至：{save_path}")

def main():
    """主函数 - 生成所有可视化"""
    print("🎨 Protocol-algorithm v2.0 - 可视化生成")
    print("=" * 50)
    
    # 生成数据
    n_nodes = 100
    positions = generate_node_positions(n_nodes)
    cluster_heads = select_cluster_heads(positions, p=0.05)
    base_station = (50.0, 150.0)
    
    # Matplotlib 静态图
    print("\n📊 生成 Matplotlib 静态图...")
    plot_network_matplotlib(positions, cluster_heads, base_station)
    
    # 生成仿真数据
    rounds = list(range(100))
    alive = [n_nodes - int(r * 0.3) for r in rounds]  # 模拟数据
    plot_energy_curve(rounds, alive)
    
    # Plotly 交互图
    print("\n🌐 生成 Plotly 交互图...")
    plot_network_plotly(positions, cluster_heads, base_station)
    
    print("\n" + "=" * 50)
    print("✅ 所有可视化已生成完成！")
    print("\n文件列表:")
    print("  - network_viz.png (网络拓扑图 - 静态)")
    print("  - energy_curve.png (能量曲线图)")
    print("  - network_interactive.html (网络拓扑 - 交互式)")

if __name__ == '__main__':
    main()
