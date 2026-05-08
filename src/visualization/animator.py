"""动态可视化动画（优化版）"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyBboxPatch
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
from dataclasses import dataclass

from ..models.network import Network


@dataclass
class AnimatorConfig:
    """动画配置"""
    figsize: Tuple[int, int] = (12, 10)
    interval: int = 200
    node_size: int = 30
    cluster_head_size: int = 100
    base_station_size: int = 200
    cluster_radius: float = 25.0
    show_grid: bool = True
    show_legend: bool = True
    color_alive: str = '#2ecc71'
    color_dead: str = '#95a5a6'
    color_cluster_head: str = '#e74c3c'
    color_base_station: str = '#3498db'
    color_cluster: str = '#9b59b6'


class OptimizedNetworkAnimator:
    """优化的网络动画"""
    
    def __init__(
        self,
        network: Network,
        config: Optional[AnimatorConfig] = None,
        save_path: Optional[str] = None
    ):
        """
        初始化优化动画器
        
        Args:
            network: 网络对象
            config: 动画配置
            save_path: 保存路径
        """
        self.network = network
        self.config = config or AnimatorConfig()
        self.save_path = save_path
        
        self._setup_figure()
        self._setup_artists()
        self._setup_cache()
    
    def _setup_figure(self) -> None:
        """设置图形"""
        self.fig, self.ax = plt.subplots(figsize=self.config.figsize)
        self.fig.patch.set_facecolor('#f8f9fa')
        
        x_min, x_max, y_min, y_max = self.network.area
        margin = 5
        
        self.ax.set_xlim(x_min - margin, x_max + margin)
        self.ax.set_ylim(y_min - margin, y_max + margin)
        self.ax.set_xlabel('X Position (m)', fontsize=12)
        self.ax.set_ylabel('Y Position (m)', fontsize=12)
        self.ax.set_aspect('equal')
        
        if self.config.show_grid:
            self.ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        self.ax.set_facecolor('#ffffff')
    
    def _setup_artists(self) -> None:
        """初始化所有 artists"""
        self.scatter_alive = self.ax.scatter(
            [], [], c=self.config.color_alive,
            s=self.config.node_size, marker='o', zorder=3
        )
        
        self.scatter_dead = self.ax.scatter(
            [], [], c=self.config.color_dead,
            s=self.config.node_size * 0.7, marker='x', zorder=2, alpha=0.5
        )
        
        self.scatter_cluster_heads = self.ax.scatter(
            [], [], c=self.config.color_cluster_head,
            s=self.config.cluster_head_size, marker='*', zorder=5
        )
        
        self.scatter_base_station = self.ax.scatter(
            [], [],
            c=self.config.color_base_station,
            s=self.config.base_station_size,
            marker='s', zorder=6, edgecolors='white', linewidths=2
        )
        
        self.text_round = self.ax.text(
            0.02, 0.98, '',
            transform=self.ax.transAxes,
            fontsize=14, fontweight='bold',
            verticalalignment='top'
        )
        
        self.text_stats = self.ax.text(
            0.02, 0.92, '',
            transform=self.ax.transAxes,
            fontsize=10, verticalalignment='top'
        )
        
        self.cluster_patches: List[Circle] = []
        
        if self.config.show_legend:
            self._setup_legend()
    
    def _setup_legend(self) -> None:
        """设置图例"""
        from matplotlib.lines import Line2D
        
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', 
                   markerfacecolor=self.config.color_alive,
                   markersize=8, label='Alive Node'),
            Line2D([0], [0], marker='*', color='w',
                   markerfacecolor=self.config.color_cluster_head,
                   markersize=12, label='Cluster Head'),
            Line2D([0], [0], marker='s', color='w',
                   markerfacecolor=self.config.color_base_station,
                   markersize=10, label='Base Station'),
            Line2D([0], [0], marker='x', color=self.config.color_dead,
                   markersize=8, linestyle='', label='Dead Node'),
        ]
        
        self.ax.legend(
            handles=legend_elements,
            loc='upper right',
            framealpha=0.9,
            fontsize=9
        )
    
    def _setup_cache(self) -> None:
        """设置缓存"""
        self._last_alive_positions: Optional[np.ndarray] = None
        self._last_dead_positions: Optional[np.ndarray] = None
        self._last_ch_positions: Optional[np.ndarray] = None
    
    def _update_alive_nodes(self, alive: List) -> None:
        """更新存活节点"""
        positions = np.array([(n.x, n.y) for n in alive])
        self.scatter_alive.set_offsets(positions)
        self._last_alive_positions = positions
    
    def _update_dead_nodes(self, dead: List) -> None:
        """更新死亡节点"""
        positions = np.array([(n.x, n.y) for n in dead])
        self.scatter_dead.set_offsets(positions)
        self._last_dead_positions = positions
    
    def _update_cluster_heads(self, cluster_heads: List) -> None:
        """更新簇头"""
        positions = np.array([(ch.node.x, ch.node.y) for ch in cluster_heads])
        self.scatter_cluster_heads.set_offsets(positions)
        self._last_ch_positions = positions
    
    def _update_clusters(self, cluster_heads: List) -> None:
        """更新簇区域"""
        for patch in self.cluster_patches:
            patch.remove()
        self.cluster_patches.clear()
        
        for ch in cluster_heads:
            if ch.node.is_alive:
                circle = Circle(
                    (ch.node.x, ch.node.y),
                    self.config.cluster_radius,
                    fill=True,
                    facecolor=self.config.color_cluster,
                    alpha=0.1,
                    edgecolor=self.config.color_cluster,
                    linestyle='--',
                    linewidth=1.5,
                    zorder=1
                )
                self.ax.add_patch(circle)
                self.cluster_patches.append(circle)
                
                for member in ch.members:
                    self.ax.plot(
                        [ch.node.x, member.x],
                        [ch.node.y, member.y],
                        color=self.config.color_cluster,
                        alpha=0.3,
                        linewidth=0.5,
                        zorder=1
                    )
    
    def _update_text(self, round_num: int) -> None:
        """更新文本"""
        alive = len(self.network.alive_nodes)
        total = self.network.n_nodes
        n_ch = len(self.network.cluster_heads)
        
        self.text_round.set_text(f'Round: {round_num}')
        
        self.text_stats.set_text(
            f'Alive: {alive}/{total} | '
            f'Cluster Heads: {n_ch} | '
            f'Energy: {self.network.total_energy:.4f} J'
        )
    
    def _setup_frame(self, frame: int) -> None:
        """设置帧"""
        self._setup_axes()
    
    def _setup_axes(self) -> None:
        """设置坐标轴"""
        x_min, x_max, y_min, y_max = self.network.area
        margin = 5
        self.ax.set_xlim(x_min - margin, x_max + margin)
        self.ax.set_ylim(y_min - margin, y_max + margin)
    
    def _plot_frame(self, frame: int) -> List:
        """绘制单帧（使用 blitting）"""
        alive = self.network.alive_nodes
        dead = self.network.dead_nodes
        
        self._update_alive_nodes(alive)
        self._update_dead_nodes(dead)
        self._update_cluster_heads(self.network.cluster_heads)
        self._update_clusters(self.network.cluster_heads)
        
        bs = self.network.base_station
        self.scatter_base_station.set_offsets([[bs.x, bs.y]])
        
        self._update_text(self.network.current_round)
        
        self._setup_frame(frame)
        
        return [
            self.scatter_alive,
            self.scatter_dead,
            self.scatter_cluster_heads,
            self.scatter_base_station,
            self.text_round,
            self.text_stats
        ]
    
    def animate(
        self,
        rounds: int,
        protocol_name: str = "leach",
        blit: bool = True,
        **kwargs
    ) -> animation.FuncAnimation:
        """
        创建动画
        
        Args:
            rounds: 轮数
            protocol_name: 协议名称
            blit: 是否使用 blitting
            **kwargs: 协议参数
            
        Returns:
            动画对象
        """
        self.network.reset()
        
        frames_data = []
        for r in range(rounds):
            self.network.simulate_round(protocol_name, **kwargs)
            frames_data.append(r)
        
        anim = animation.FuncAnimation(
            self.fig,
            self._plot_frame,
            frames=frames_data,
            interval=self.config.interval,
            blit=blit,
            cache_frame_data=not blit
        )
        
        return anim
    
    def animate_to_list(
        self,
        rounds: int,
        protocol_name: str = "leach",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        预计算所有帧数据（用于快速预览）
        
        Args:
            rounds: 轮数
            protocol_name: 协议名称
            **kwargs: 协议参数
            
        Returns:
            每帧的数据列表
        """
        self.network.reset()
        
        frames = []
        for r in range(rounds):
            self.network.simulate_round(protocol_name, **kwargs)
            
            frame_data = {
                'round': r,
                'alive_nodes': [
                    {'id': n.id, 'x': n.x, 'y': n.y, 'energy': n.energy}
                    for n in self.network.alive_nodes
                ],
                'dead_nodes': [
                    {'id': n.id, 'x': n.x, 'y': n.y}
                    for n in self.network.dead_nodes
                ],
                'cluster_heads': [
                    {
                        'id': ch.node.id,
                        'x': ch.node.x,
                        'y': ch.node.y,
                        'n_members': ch.n_members
                    }
                    for ch in self.network.cluster_heads
                ],
                'total_energy': self.network.total_energy,
                'n_alive': self.network.n_alive
            }
            frames.append(frame_data)
        
        return frames
    
    def save(
        self,
        anim: animation.FuncAnimation,
        path: str,
        fps: int = 10,
        dpi: int = 100
    ) -> None:
        """
        保存动画
        
        Args:
            anim: 动画对象
            path: 保存路径
            fps: 帧率
            dpi: 分辨率
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if save_path.suffix == '.gif':
            anim.save(path, writer='pillow', fps=fps)
        elif save_path.suffix == '.mp4':
            try:
                anim.save(path, writer='ffmpeg', fps=fps, dpi=dpi)
            except Exception as e:
                print(f"FFmpeg not available, trying HTML5 writer: {e}")
                anim.save(path, writer='html', fps=fps)
        else:
            raise ValueError(f"Unsupported format: {save_path.suffix}")
        
        print(f"Animation saved to {path}")
    
    def create_comparison_animation(
        self,
        networks: List[Network],
        rounds: int,
        protocol_names: Optional[List[str]] = None,
        titles: Optional[List[str]] = None
    ) -> animation.FuncAnimation:
        """
        创建多网络对比动画
        
        Args:
            networks: 网络列表
            rounds: 轮数
            protocol_names: 协议名称列表
            titles: 标题列表
            
        Returns:
            动画对象
        """
        n_networks = len(networks)
        protocol_names = protocol_names or ['leach'] * n_networks
        titles = titles or [f'Network {i+1}' for i in range(n_networks)]
        
        cols = min(2, n_networks)
        rows = (n_networks + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(10 * cols, 8 * rows))
        axes = np.atleast_2d(axes)
        
        fig.suptitle('LEACH Protocol Comparison', fontsize=16, fontweight='bold')
        
        animators = []
        scatters_alive = []
        scatters_dead = []
        scatters_ch = []
        texts = []
        
        for i, (network, protocol, title) in enumerate(zip(networks, protocol_names, titles)):
            row, col = i // cols, i % cols
            ax = axes[row, col]
            
            x_min, x_max, y_min, y_max = network.area
            ax.set_xlim(x_min - 5, x_max + 5)
            ax.set_ylim(y_min - 5, y_max + 5)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
            sc_alive = ax.scatter([], [], c='green', s=20, zorder=3)
            sc_dead = ax.scatter([], [], c='gray', s=15, marker='x', zorder=2)
            sc_ch = ax.scatter([], [], c='red', s=80, marker='*', zorder=4)
            
            txt = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=9,
                         verticalalignment='top')
            
            animators.append(network)
            scatters_alive.append(sc_alive)
            scatters_dead.append(sc_dead)
            scatters_ch.append(sc_ch)
            texts.append(txt)
        
        for network in networks:
            network.reset()
        
        for _ in range(rounds):
            for network, protocol in zip(networks, protocol_names):
                network.simulate_round(protocol)
        
        for network in networks:
            network.reset()
        
        def update(frame):
            for i, (network, sc_alive, sc_dead, sc_ch, txt) in enumerate(
                zip(animators, scatters_alive, scatters_dead, scatters_ch, texts)
            ):
                network.simulate_round(protocol_names[i])
                
                alive = network.alive_nodes
                dead = network.dead_nodes
                
                sc_alive.set_offsets([(n.x, n.y) for n in alive])
                sc_dead.set_offsets([(n.x, n.y) for n in dead])
                sc_ch.set_offsets([
                    (ch.node.x, ch.node.y)
                    for ch in network.cluster_heads
                ])
                
                txt.set_text(f'R:{network.current_round} A:{len(alive)}')
            
            return sum([[s, t] for s, t in zip(scatters_alive + scatters_dead + scatters_ch, texts)], [])
        
        return animation.FuncAnimation(
            fig, update,
            frames=rounds,
            interval=200,
            blit=False
        )
    
    def close(self) -> None:
        """关闭图形"""
        plt.close(self.fig)
