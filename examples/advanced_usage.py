#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""高级用法示例

展示 LEACH 仿真平台的高级功能和用法,包括:
- 并行仿真实验
- 参数扫描与优化
- 自定义协议扩展
- 批量仿真与结果分析
- 网络拓扑可视化
- 能量模型调参
- 协议注册与插件化扩展

用法:
    python examples/advanced_usage.py
    python examples/advanced_usage.py --demo parallel
    python examples/advanced_usage.py --demo parameter_sweep
    python examples/advanced_usage.py --demo custom_protocol
"""

from __future__ import annotations

import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.network import Network
from src.models.node import Node, NodeRole
from src.models.cluster_head import ClusterHead
from src.energy.radio_model import FirstOrderRadioModel
from src.leach.base import LEACHProtocol
from src.leach.variants import LEACHRegistry
from src.simulation.engine import (
    ParallelSimulationEngine,
    SimulationConfig,
    BatchSimulator,
)
from src.visualization.metrics_plots import MetricsPlotter
from src.visualization.comparison import ComparisonPlotter


# =========================================================================
# 示例 1: 并行仿真实验
# =========================================================================

def demo_parallel_simulation():
    """
    演示并行仿真实验

    使用多进程并行运行多个仿真配置, 加速批量实验。
    """
    print("\n" + "=" * 70)
    print("示例 1: 并行仿真实验")
    print("=" * 70)

    engine = ParallelSimulationEngine(n_workers=2)

    # 定义要对比的配置
    configs = [
        SimulationConfig(
            n_nodes=50,
            area=(0, 100, 0, 100),
            base_station_pos=(50, 50),
            initial_energy=0.5,
            rounds=500,
            protocol_name="leach",
            seed=42,
        ),
        SimulationConfig(
            n_nodes=100,
            area=(0, 100, 0, 100),
            base_station_pos=(50, 50),
            initial_energy=0.5,
            rounds=500,
            protocol_name="leach",
            seed=42,
        ),
        SimulationConfig(
            n_nodes=200,
            area=(0, 100, 0, 100),
            base_station_pos=(50, 50),
            initial_energy=0.5,
            rounds=500,
            protocol_name="leach",
            seed=42,
        ),
    ]

    print(f"运行 {len(configs)} 个并行仿真...")
    print(f"工作进程数: {engine.n_workers}")

    start_time = time.time()
    results = engine.run_parallel_experiments(configs, show_progress=True)
    elapsed = time.time() - start_time

    print(f"\n并行仿真完成, 耗时: {elapsed:.2f}s")
    print()

    for result in results:
        cfg = result.config
        res = result.results
        print(
            f"n_nodes={cfg.n_nodes:>3}: "
            f"生命周期={res['network_lifetime']:>5} 轮, "
            f"半生命周期={res['half_dead_round']:>5} 轮, "
            f"耗时={result.execution_time:.2f}s"
        )


# =========================================================================
# 示例 2: 参数扫描
# =========================================================================

def demo_parameter_sweep():
    """
    演示参数扫描实验

    自动扫描多个参数组合, 找到最优配置。
    """
    print("\n" + "=" * 70)
    print("示例 2: 参数扫描实验")
    print("=" * 70)

    engine = ParallelSimulationEngine(n_workers=2)

    # 基础配置
    base_config = SimulationConfig(
        n_nodes=100,
        area=(0, 100, 0, 100),
        base_station_pos=(50, 50),
        initial_energy=0.5,
        rounds=500,
        protocol_name="leach",
        seed=42,
    )

    # 参数网格
    param_grid = {
        "n_nodes": [50, 100, 150],
    }

    print("参数网格:")
    for name, values in param_grid.items():
        print(f"  - {name}: {values}")
    print()

    # 运行参数扫描
    print("运行参数扫描...")
    start_time = time.time()
    results = engine.run_parameter_sweep(
        base_config=base_config,
        param_grid=param_grid,
        n_runs=2,
        aggregate=True
    )
    elapsed = time.time() - start_time

    print(f"参数扫描完成, 耗时: {elapsed:.2f}s")
    print()

    # 打印汇总结果
    if "by_params" in results:
        print("汇总结果:")
        for key, data in results["by_params"].items():
            lt = data.get("network_lifetime", {})
            print(
                f"  {key}: "
                f"平均生命周期={lt.get('mean', 0):.1f} ± {lt.get('std', 0):.1f} 轮"
            )


# =========================================================================
# 示例 3: 自定义协议扩展
# =========================================================================

class LEACHWeighted(LEACHProtocol):
    """
    自定义协议: 加权 LEACH

    结合节点剩余能量和到基站的距离, 加权选择簇头。
    权重: 70% 能量 + 30% 距离 (距离越近权重越高)
    """

    def __init__(self, p: float = 0.05, energy_weight: float = 0.7):
        """
        初始化

        Args:
            p: 簇头概率
            energy_weight: 能量权重 (0~1)
        """
        super().__init__(p)
        self.energy_weight = energy_weight
        self.distance_weight = 1.0 - energy_weight

    def select_cluster_heads(
        self, network: "Network", **kwargs
    ) -> List[ClusterHead]:
        """
        基于加权分数选择簇头

        Args:
            network: 网络对象

        Returns:
            簇头列表
        """
        alive_nodes = network.alive_nodes
        n_nodes = len(alive_nodes)
        n_clusters = max(1, int(n_nodes * self.p))

        bs_pos = network.base_station.position
        max_dist = 100.0  # 最大可能距离

        # 计算每个节点的加权分数
        node_scores = []
        for node in alive_nodes:
            energy_ratio = node.energy / node.initial_energy
            dist_ratio = 1.0 - (node.distance_to(bs_pos) / max_dist)

            score = (
                self.energy_weight * energy_ratio +
                self.distance_weight * dist_ratio
            )
            node_scores.append((node, score))

        # 按分数降序排序, 选择前 n_clusters 个
        node_scores.sort(key=lambda x: x[1], reverse=True)

        cluster_id = 0
        cluster_heads = []

        for node, score in node_scores[:n_clusters]:
            node.become_cluster_head(cluster_id)

            ch = ClusterHead(
                node=node,
                cluster_id=cluster_id
            )
            cluster_heads.append(ch)
            cluster_id += 1

        self.current_round += 1

        return cluster_heads


def demo_custom_protocol():
    """
    演示自定义协议的使用

    1. 注册自定义协议到注册表
    2. 使用注册表运行仿真
    3. 对比自定义协议与经典 LEACH
    """
    print("\n" + "=" * 70)
    print("示例 3: 自定义协议扩展")
    print("=" * 70)

    # 注册自定义协议
    LEACHRegistry.register("leach_weighted", LEACHWeighted)
    print("已注册自定义协议: leach_weighted")
    print(f"可用协议: {LEACHRegistry.list_protocols()}")
    print()

    # 对比实验
    protocols = ["leach", "leach_weighted"]
    results_list = []

    for protocol in protocols:
        print(f"运行 {protocol}...")

        network = Network(
            n_nodes=100,
            area=(0, 100, 0, 100),
            base_station_pos=(50, 50),
            initial_energy=0.5,
            seed=42,
        )

        results = network.simulate_network(
            rounds=1000,
            protocol_name=protocol
        )
        results_list.append(results)

        print(
            f"  生命周期={results['network_lifetime']} 轮, "
            f"半生命周期={results['half_dead_round']} 轮"
        )

    # 生成对比图
    output_dir = "results/advanced/custom_protocol"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    plotter = ComparisonPlotter()
    plotter.compare_protocols(
        results_list=results_list,
        protocol_names=["LEACH (经典)", "LEACH-Weighted (加权)"],
        save_path=f"{output_dir}/custom_protocol_comparison.png",
        show=False
    )

    print(f"\n对比图已保存到 {output_dir}/")


# =========================================================================
# 示例 4: 批量仿真器
# =========================================================================

def demo_batch_simulator():
    """
    演示批量仿真器的使用

    一键对比多个协议。
    """
    print("\n" + "=" * 70)
    print("示例 4: 批量仿真器")
    print("=" * 70)

    batch = BatchSimulator()

    print("对比协议: leach, leach-c, leach-ee")
    print("节点数: 100, 轮数: 500, 重复: 2 次")
    print()

    comparison = batch.compare_protocols(
        n_nodes=100,
        area=(0, 100, 0, 100),
        rounds=500,
        protocols=["leach", "leach-c", "leach-ee"],
        n_runs=2
    )

    # 打印对比结果
    summary = comparison.get("summary", {})
    lifetimes = summary.get("lifetimes", {})
    best = summary.get("best_protocol", "N/A")

    print("\n对比结果:")
    print(f"{'协议':<15} {'平均生命周期':>15}")
    print("-" * 32)
    for protocol, lifetime in lifetimes.items():
        print(f"{protocol:<15} {lifetime:>15.1f}")
    print(f"\n最优协议: {best}")

    # 保存结果
    output_dir = "results/advanced/batch"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    batch.save_results(comparison, f"{output_dir}/batch_comparison.json")

    print(f"结果已保存到 {output_dir}/")


# =========================================================================
# 示例 5: 能量模型调参
# =========================================================================

def demo_energy_model_tuning():
    """
    演示能量模型参数调优

    不同的能量模型参数对仿真结果的影响。
    """
    print("\n" + "=" * 70)
    print("示例 5: 能量模型调参")
    print("=" * 70)

    # 测试不同的 E_elec 值
    e_elec_values = [25e-9, 50e-9, 100e-9]

    results_list = []
    protocol_names = []

    for e_elec in e_elec_values:
        print(f"\n测试 E_elec = {e_elec:.0e} J/bit...")

        energy_model = FirstOrderRadioModel(E_elec=e_elec)

        network = Network(
            n_nodes=100,
            area=(0, 100, 0, 100),
            base_station_pos=(50, 50),
            energy_model=energy_model,
            initial_energy=0.5,
            seed=42,
        )

        results = network.simulate_network(
            rounds=1000,
            protocol_name="leach"
        )
        results_list.append(results)
        protocol_names.append(f"E_elec={e_elec:.0e}")

        print(
            f"  生命周期={results['network_lifetime']} 轮, "
            f"剩余能量={results['final_energy']:.4f} J"
        )

    # 生成对比图
    output_dir = "results/advanced/energy_tuning"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    plotter = ComparisonPlotter()
    plotter.compare_protocols(
        results_list=results_list,
        protocol_names=protocol_names,
        save_path=f"{output_dir}/energy_model_tuning.png",
        show=False
    )

    print(f"\n对比图已保存到 {output_dir}/")


# =========================================================================
# 示例 6: 网络拓扑快照
# =========================================================================

def demo_network_topology_snapshot():
    """
    演示网络拓扑快照功能

    在仿真过程中保存特定轮次的网络拓扑状态。
    """
    print("\n" + "=" * 70)
    print("示例 6: 网络拓扑快照")
    print("=" * 70)

    network = Network(
        n_nodes=100,
        area=(0, 100, 0, 100),
        base_station_pos=(50, 50),
        initial_energy=0.5,
        seed=42,
    )

    # 在特定轮次保存快照
    snapshot_rounds = [1, 100, 500, 1000]

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for r in range(max(snapshot_rounds)):
        network.simulate_round(protocol_name="leach")

        if (r + 1) in snapshot_rounds:
            idx = snapshot_rounds.index(r + 1)
            ax = axes[idx]

            # 绘制节点
            for node in network.alive_nodes:
                if node.is_cluster_head:
                    ax.scatter(
                        node.x, node.y,
                        c='red', s=100, marker='*',
                        label='CH' if idx == 0 else None
                    )
                else:
                    ax.scatter(
                        node.x, node.y,
                        c='blue', s=20, alpha=0.6,
                        label='Normal' if idx == 0 else None
                    )

            # 绘制死亡节点
            for node in network.dead_nodes:
                ax.scatter(
                    node.x, node.y,
                    c='gray', s=20, alpha=0.3,
                    marker='x',
                    label='Dead' if idx == 0 else None
                )

            # 绘制基站
            ax.scatter(
                network.base_station.position[0],
                network.base_station.position[1],
                c='green', s=200, marker='s',
                label='BS' if idx == 0 else None
            )

            ax.set_title(f"Round {r + 1}")
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.set_aspect('equal')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_dir = "results/advanced/topology"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{output_dir}/topology_snapshots.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"拓扑快照已保存到 {output_dir}/topology_snapshots.png")


# =========================================================================
# 示例 7: 节点能量分布分析
# =========================================================================

def demo_energy_distribution_analysis():
    """
    演示节点能量分布分析

    分析仿真过程中节点能量的分布变化。
    """
    print("\n" + "=" * 70)
    print("示例 7: 节点能量分布分析")
    print("=" * 70)

    network = Network(
        n_nodes=100,
        area=(0, 100, 0, 100),
        base_station_pos=(50, 50),
        initial_energy=0.5,
        seed=42,
    )

    # 在特定轮次记录能量分布
    analysis_rounds = [1, 200, 500, 1000]
    energy_snapshots = []

    for r in range(max(analysis_rounds)):
        network.simulate_round(protocol_name="leach")

        if (r + 1) in analysis_rounds:
            energies = [n.energy for n in network.nodes]
            energy_snapshots.append({
                "round": r + 1,
                "mean": np.mean(energies),
                "std": np.std(energies),
                "min": np.min(energies),
                "max": np.max(energies),
                "dead": sum(1 for e in energies if e <= 0),
            })

    # 打印分析结果
    print("\n能量分布分析:")
    print(f"{'轮次':>6} {'平均能量':>12} {'标准差':>10} {'最小值':>10} {'最大值':>10} {'死亡数':>6}")
    print("-" * 60)
    for snap in energy_snapshots:
        print(
            f"{snap['round']:>6} "
            f"{snap['mean']:>12.4f} "
            f"{snap['std']:>10.4f} "
            f"{snap['min']:>10.4f} "
            f"{snap['max']:>10.4f} "
            f"{snap['dead']:>6}"
        )


# =========================================================================
# 主函数
# =========================================================================

def parse_args():
    """解析命令行参数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="高级用法示例",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--demo",
        type=str,
        default="all",
        choices=[
            "parallel",
            "parameter_sweep",
            "custom_protocol",
            "batch_simulator",
            "energy_tuning",
            "topology",
            "energy_distribution",
            "all",
        ],
        help="要运行的演示 (默认: all)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    demos = {
        "parallel": demo_parallel_simulation,
        "parameter_sweep": demo_parameter_sweep,
        "custom_protocol": demo_custom_protocol,
        "batch_simulator": demo_batch_simulator,
        "energy_tuning": demo_energy_model_tuning,
        "topology": demo_network_topology_snapshot,
        "energy_distribution": demo_energy_distribution_analysis,
    }

    if args.demo == "all":
        for name, func in demos.items():
            try:
                func()
            except Exception as e:
                print(f"\n演示 {name} 失败: {e}")
    else:
        demos[args.demo]()
