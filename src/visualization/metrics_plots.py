"""指标图表绘制"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any
from pathlib import Path

from ..models.network import Network


class MetricsPlotter:
    """指标图表绘制"""
    
    def __init__(self, style: str = "seaborn-v0_8-darkgrid"):
        """
        初始化
        
        Args:
            style: 绘图风格
        """
        plt.style.use(style)
        self.colors = plt.cm.Set2(np.linspace(0, 1, 8))
    
    def plot_network_lifetime(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        绘制网络生命周期
        
        Args:
            results: 仿真结果
            save_path: 保存路径
            show: 是否显示
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        rounds = results['rounds']
        alive = results['alive_nodes']
        n_nodes = results['alive_nodes'][0] + results['dead_nodes'][0]
        
        ax.plot(rounds, alive, color=self.colors[0], linewidth=2, label='Alive Nodes')
        ax.axhline(y=n_nodes / 2, color='red', linestyle='--', alpha=0.7, label='Half Dead')
        ax.axhline(y=n_nodes / 10, color='orange', linestyle='--', alpha=0.7, label='10% Alive')
        
        # 标记第一个死亡节点
        first_dead = results.get('first_dead_round')
        if first_dead and first_dead < len(alive):
            ax.axvline(x=first_dead, color='gray', linestyle=':', alpha=0.5)
            ax.annotate(
                f'First Dead\nRound {first_dead}',
                xy=(first_dead, alive[first_dead]),
                xytext=(first_dead + 50, alive[first_dead] + 10),
                arrowprops=dict(arrowstyle='->', color='gray'),
            )
        
        ax.set_xlabel('Round')
        ax.set_ylabel('Number of Alive Nodes')
        ax.set_title('Network Lifetime')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_energy_consumption(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        绘制能量消耗
        
        Args:
            results: 仿真结果
            save_path: 保存路径
            show: 是否显示
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        rounds = results['rounds']
        
        # 总能耗
        ax1 = axes[0]
        ax1.plot(rounds, results['total_energy'], color=self.colors[1], linewidth=2)
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Total Energy (J)')
        ax1.set_title('Total Energy Consumption')
        ax1.grid(True, alpha=0.3)
        
        # 能量变化率
        ax2 = axes[1]
        if len(rounds) > 1:
            energy_diff = np.diff(results['total_energy'])
            ax2.bar(rounds[1:], -energy_diff, color=self.colors[2], alpha=0.7)
            ax2.set_xlabel('Round')
            ax2.set_ylabel('Energy Consumed (J)')
            ax2.set_title('Energy Consumption per Round')
            ax2.grid(True, alpha=0.3)
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_cluster_distribution(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        绘制簇分布
        
        Args:
            results: 仿真结果
            save_path: 保存路径
            show: 是否显示
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        rounds = results['rounds']
        n_chs = results['cluster_heads']
        
        ax.bar(rounds, n_chs, color=self.colors[3], alpha=0.7)
        ax.axhline(y=np.mean(n_chs), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(n_chs):.1f}')
        
        ax.set_xlabel('Round')
        ax.set_ylabel('Number of Cluster Heads')
        ax.set_title('Cluster Head Distribution per Round')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_all_metrics(
        self,
        results: Dict[str, Any],
        output_dir: str,
        show: bool = False
    ):
        """
        绘制所有指标
        
        Args:
            results: 仿真结果
            output_dir: 输出目录
            show: 是否显示
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.plot_network_lifetime(
            results,
            save_path=str(output_path / 'network_lifetime.png'),
            show=show
        )
        
        self.plot_energy_consumption(
            results,
            save_path=str(output_path / 'energy_consumption.png'),
            show=show
        )
        
        self.plot_cluster_distribution(
            results,
            save_path=str(output_path / 'cluster_distribution.png'),
            show=show
        )
        
        print(f"Plots saved to {output_dir}")
