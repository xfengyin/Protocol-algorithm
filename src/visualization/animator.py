"""动态可视化动画"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Optional, Tuple
from pathlib import Path

from ..models.network import Network


class NetworkAnimator:
    """网络动态可视化"""
    
    def __init__(
        self,
        network: Network,
        figsize: Tuple[int, int] = (10, 10),
        interval: int = 200,
        save_path: Optional[str] = None
    ):
        """
        初始化
        
        Args:
            network: 网络对象
            figsize: 图形大小
            interval: 帧间隔 (ms)
            save_path: 保存路径
        """
        self.network = network
        self.figsize = figsize
        self.interval = interval
        self.save_path = save_path
        
        self.fig, self.ax = plt.subplots(figsize=figsize)
    
    def _setup_axes(self):
        """设置坐标轴"""
        self.ax.clear()
        x_min, x_max, y_min, y_max = self.network.area
        self.ax.set_xlim(x_min - 5, x_max + 5)
        self.ax.set_ylim(y_min - 5, y_max + 5)
        self.ax.set_xlabel('X Position (m)')
        self.ax.set_ylabel('Y Position (m)')
        self.ax.set_title('LEACH Network Simulation')
        self.ax.grid(True, alpha=0.3)
    
    def _plot_round(self, frame):
        """绘制当前轮"""
        self._setup_axes()
        
        # 绘制基站
        bs = self.network.base_station
        self.ax.scatter(bs.x, bs.y, marker='s', s=200, c='red', zorder=5, label='Base Station')
        
        # 分类节点
        alive = self.network.alive_nodes
        dead = self.network.dead_nodes
        
        # 绘制存活节点
        if alive:
            for node in alive:
                if node.is_cluster_head:
                    color = 'blue'
                    marker = '*'
                    size = 100
                else:
                    color = 'green'
                    marker = 'o'
                    size = 30
                
                self.ax.scatter(node.x, node.y, c=color, marker=marker, s=size)
                
                # 如果是簇头，绘制簇边界
                if node.is_cluster_head and node.cluster_id is not None:
                    circle = plt.Circle(
                        (node.x, node.y),
                        30,
                        fill=False,
                        color='blue',
                        linestyle='--',
                        alpha=0.5
                    )
                    self.ax.add_patch(circle)
        
        # 绘制死亡节点
        if dead:
            self.ax.scatter(
                [n.x for n in dead],
                [n.y for n in dead],
                c='gray',
                marker='x',
                s=20,
                alpha=0.5
            )
        
        # 更新标题
        self.ax.set_title(
            f'Round {self.network.current_round} | '
            f'Alive: {len(alive)}/{self.network.n_nodes}'
        )
        
        if frame == 0:
            self.ax.legend()
        
        return []
    
    def animate(
        self,
        rounds: int,
        protocol_name: str = "leach",
        **kwargs
    ) -> animation.FuncAnimation:
        """
        创建动画
        
        Args:
            rounds: 轮数
            protocol_name: 协议名称
            **kwargs: 协议参数
            
        Returns:
            动画对象
        """
        self.network.reset()
        
        frames = []
        for _ in range(rounds):
            self.network.simulate_round(protocol_name, **kwargs)
            frames.append(_)
        
        anim = animation.FuncAnimation(
            self.fig,
            self._plot_round,
            frames=len(frames),
            interval=self.interval,
            blit=False
        )
        
        return anim
    
    def save(self, anim: animation.FuncAnimation, path: str, fps: int = 10):
        """
        保存动画
        
        Args:
            anim: 动画对象
            path: 保存路径
            fps: 帧率
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if save_path.suffix == '.gif':
            anim.save(path, writer='pillow', fps=fps)
        elif save_path.suffix == '.mp4':
            anim.save(path, writer='ffmpeg', fps=fps)
        else:
            raise ValueError(f"Unsupported format: {save_path.suffix}")
        
        print(f"Animation saved to {path}")
