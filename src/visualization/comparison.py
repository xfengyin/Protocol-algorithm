"""对比实验图表"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
from pathlib import Path


class ComparisonPlotter:
    """对比实验图表"""
    
    def __init__(self):
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        self.linestyles = ['-', '--', '-.', ':', '-']
    
    def compare_protocols(
        self,
        results_list: List[Dict[str, Any]],
        protocol_names: List[str],
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        对比多个协议
        
        Args:
            results_list: 结果列表
            protocol_names: 协议名称
            save_path: 保存路径
            show: 是否显示
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 网络生命周期
        ax1 = axes[0, 0]
        for i, (results, name) in enumerate(zip(results_list, protocol_names)):
            ax1.plot(
                results['rounds'],
                results['alive_nodes'],
                color=self.colors[i % len(self.colors)],
                linestyle=self.linestyles[i % len(self.linestyles)],
                linewidth=2,
                label=name
            )
        
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Alive Nodes')
        ax1.set_title('Network Lifetime Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 总能耗
        ax2 = axes[0, 1]
        for i, (results, name) in enumerate(zip(results_list, protocol_names)):
            ax2.plot(
                results['rounds'],
                results['total_energy'],
                color=self.colors[i % len(self.colors)],
                linestyle=self.linestyles[i % len(self.linestyles)],
                linewidth=2,
                label=name
            )
        
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Total Energy (J)')
        ax2.set_title('Total Energy Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 性能指标对比
        ax3 = axes[1, 0]
        self._plot_performance_comparison(results_list, protocol_names, ax3)
        
        # 存活率曲线
        ax4 = axes[1, 1]
        for i, (results, name) in enumerate(zip(results_list, protocol_names)):
            if results['alive_nodes']:
                n_initial = results['alive_nodes'][0] + results['dead_nodes'][0]
                survival_rate = np.array(results['alive_nodes']) / n_initial * 100
                ax4.plot(
                    results['rounds'],
                    survival_rate,
                    color=self.colors[i % len(self.colors)],
                    linestyle=self.linestyles[i % len(self.linestyles)],
                    linewidth=2,
                    label=name
                )
        
        ax4.set_xlabel('Round')
        ax4.set_ylabel('Survival Rate (%)')
        ax4.set_title('Node Survival Rate')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 105)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def _plot_performance_comparison(
        self,
        results_list: List[Dict[str, Any]],
        protocol_names: List[str],
        ax
    ):
        """绘制性能指标对比"""
        metrics = []
        
        for results in results_list:
            alive = np.array(results['alive_nodes'])
            rounds = np.array(results['rounds'])
            
            first_dead = np.argmax(alive < alive[0] + alive[0])
            if alive[first_dead] == alive[0] + alive[0]:
                first_dead = len(alive)
            
            half_dead = np.argmax(alive <= (alive[0] + alive[0]) / 2)
            if alive[half_dead] > (alive[0] + alive[0]) / 2:
                half_dead = len(alive)
            
            metrics.append({
                'First Dead': first_dead,
                'Half Dead': half_dead,
                'Total Rounds': len(rounds),
            })
        
        x = np.arange(len(metrics[0]))
        width = 0.25
        
        for i, (m, name) in enumerate(zip(metrics, protocol_names)):
            values = list(m.values())
            ax.bar(x + i * width, values, width, 
                   color=self.colors[i % len(self.colors)],
                   label=name, alpha=0.8)
        
        ax.set_ylabel('Rounds')
        ax.set_title('Performance Metrics')
        ax.set_xticks(x + width * (len(metrics) - 1) / 2)
        ax.set_xticklabels(list(metrics[0].keys()))
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    def generate_comparison_report(
        self,
        results_list: List[Dict[str, Any]],
        protocol_names: List[str],
        output_dir: str
    ):
        """
        生成对比报告
        
        Args:
            results_list: 结果列表
            protocol_names: 协议名称
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 绘制对比图
        self.compare_protocols(
            results_list,
            protocol_names,
            save_path=str(output_path / 'comparison.png'),
            show=False
        )
        
        # 生成统计报告
        report_lines = ["# Protocol Comparison Report\n"]
        report_lines.append("=" * 50 + "\n\n")
        
        for results, name in zip(results_list, protocol_names):
            alive = np.array(results['alive_nodes'])
            n_initial = alive[0] + results['dead_nodes'][0]
            
            first_dead = results.get('first_dead_round', len(alive))
            half_dead = results.get('half_dead_round', len(alive))
            
            report_lines.append(f"## {name}\n")
            report_lines.append(f"- Network Lifetime (First Death): {first_dead} rounds\n")
            report_lines.append(f"- Half Network Lifetime: {half_dead} rounds\n")
            report_lines.append(f"- Final Alive Nodes: {alive[-1]}/{n_initial}\n")
            report_lines.append(f"- Total Energy Remaining: {results['total_energy'][-1]:.4f} J\n")
            report_lines.append(f"- Average Cluster Heads: {np.mean(results['cluster_heads']):.1f}\n")
            report_lines.append("\n")
        
        report_path = output_path / 'comparison_report.md'
        report_path.write_text(''.join(report_lines))
        
        print(f"Comparison report saved to {output_dir}")
