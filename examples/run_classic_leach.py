#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""经典 LEACH 协议运行示例

演示如何使用 LEACH 协议仿真平台进行经典 LEACH 仿真,包括:
- 网络初始化与配置
- 运行仿真并收集指标
- 可视化结果图表
- 保存仿真结果

用法:
    python examples/run_classic_leach.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.network import Network
from src.energy.radio_model import FirstOrderRadioModel
from src.visualization.metrics_plots import MetricsPlotter


def run_classic_leach(
    n_nodes: int = 100,
    rounds: int = 1000,
    seed: int = 42,
    output_dir: str = "results/classic_leach"
) -> dict:
    """
    运行经典 LEACH 协议仿真

    Args:
        n_nodes: 节点数量
        rounds: 仿真轮数
        seed: 随机种子
        output_dir: 结果输出目录

    Returns:
        仿真结果字典
    """
    # =========================================================================
    # 1. 初始化能量模型
    # =========================================================================
    # First Order Radio Model 参数:
    #   E_elec=50nJ/bit     - 发射/接收电路能耗
    #   epsilon_fs=10pJ/bit - 自由空间能耗系数
    #   epsilon_mp=0.0013pJ - 多径衰落能耗系数
    #   d_threshold=87m     - 距离阈值
    #   E_da=5nJ/bit        - 数据聚合能耗
    energy_model = FirstOrderRadioModel()

    print("=" * 60)
    print("经典 LEACH 协议仿真")
    print("=" * 60)
    print(f"能量模型: {energy_model}")
    print()

    # =========================================================================
    # 2. 创建网络
    # =========================================================================
    network = Network(
        n_nodes=n_nodes,
        area=(0, 100, 0, 100),
        base_station_pos=(50, 50),
        energy_model=energy_model,
        initial_energy=0.5,
        seed=seed
    )

    print(f"网络配置:")
    print(f"  - 节点数量: {n_nodes}")
    print(f"  - 区域范围: 100m x 100m")
    print(f"  - 基站位置: (50, 50)")
    print(f"  - 初始能量: 0.5 J/节点")
    print(f"  - 随机种子: {seed}")
    print(f"  - 存活节点: {network.n_alive}")
    print()

    # =========================================================================
    # 3. 运行仿真
    # =========================================================================
    # 默认使用经典 LEACH 协议 (簇头概率 p=0.05)
    # 自定义停止条件: 当所有节点死亡时停止
    def stop_when_all_dead(net: Network, round_num: int) -> bool:
        """当所有节点死亡时停止仿真"""
        return net.n_alive == 0

    print(f"开始仿真 (最大 {rounds} 轮)...")
    results = network.simulate_network(
        rounds=rounds,
        protocol_name="leach",
        stop_condition=stop_when_all_dead
    )
    print(f"仿真完成!")
    print()

    # =========================================================================
    # 4. 打印结果摘要
    # =========================================================================
    print("=" * 60)
    print("仿真结果")
    print("=" * 60)
    print(f"  - 第一轮有节点死亡: {results['first_dead_round']} 轮")
    print(f"  - 一半节点死亡:     {results['half_dead_round']} 轮")
    print(f"  - 网络生命周期:     {results['network_lifetime']} 轮")
    print(f"  - 最终存活节点:     {results['alive_nodes'][-1]}/{n_nodes}")
    print(f"  - 最终剩余能量:     {results['final_energy']:.4f} J")
    print(f"  - 实际运行轮数:     {results['total_rounds_simulated']}")
    print()

    # =========================================================================
    # 5. 可视化
    # =========================================================================
    print("生成可视化图表...")
    plotter = MetricsPlotter()
    plotter.plot_all_metrics(
        results,
        output_dir=output_dir,
        show=False
    )
    print(f"图表已保存到 {output_dir}/")

    return results


def demo_single_round():
    """演示单轮仿真过程,展示 LEACH 协议的 setup 和 steady 阶段"""
    print("\n" + "=" * 60)
    print("单轮仿真演示")
    print("=" * 60)

    network = Network(
        n_nodes=20,
        area=(0, 50, 0, 50),
        base_station_pos=(25, 25),
        initial_energy=0.5,
        seed=42
    )

    print(f"\n初始状态: {network.n_alive} 个节点, 总能量 {network.total_energy:.4f} J")

    # 模拟前 5 轮
    for r in range(5):
        metrics = network.simulate_round(protocol_name="leach")
        print(
            f"轮 {r + 1}: "
            f"存活={metrics.alive_nodes}, "
            f"簇头={metrics.n_cluster_heads}, "
            f"总能量={metrics.total_energy:.4f} J, "
            f"平均能量={metrics.average_energy:.4f} J"
        )


def demo_custom_protocol_params():
    """演示使用自定义协议参数"""
    print("\n" + "=" * 60)
    print("自定义协议参数示例")
    print("=" * 60)

    network = Network(
        n_nodes=100,
        area=(0, 100, 0, 100),
        base_station_pos=(50, 50),
        initial_energy=0.5,
        seed=42
    )

    # 尝试不同的簇头概率
    for p in [0.03, 0.05, 0.08]:
        network.reset()
        results = network.simulate_network(
            rounds=500,
            protocol_name="leach"
        )
        print(
            f"簇头概率 p={p:.2f}: "
            f"生命周期={results['network_lifetime']} 轮, "
            f"半生命周期={results['half_dead_round']} 轮"
        )


if __name__ == "__main__":
    # 运行完整仿真
    results = run_classic_leach()

    # 运行额外演示
    demo_single_round()
    demo_custom_protocol_params()
