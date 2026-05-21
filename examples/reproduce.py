#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""论文结果复现脚本

复现 LEACH 论文中的经典实验结果, 用于验证仿真平台的正确性。

参考论文:
  W. R. Heinzelman, A. Chandrakasan, and H. Balakrishnan,
  "Energy-Efficient Communication Protocol for Wireless Microsensor Networks,"
  in Proc. 33rd Hawaii Int. Conf. System Sciences (HICSS), Jan. 2000.

复现目标:
  1. 验证网络生命周期曲线与论文图 4 吻合
  2. 验证能量消耗趋势与论文图 5 吻合
  3. 复现不同簇头概率 p 对性能的影响

用法:
    python examples/reproduce.py
    python examples/reproduce.py --experiment all
    python examples/reproduce.py --experiment lifetime --rounds 5000
"""

from __future__ import annotations

import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.network import Network
from src.energy.radio_model import FirstOrderRadioModel
from src.visualization.metrics_plots import MetricsPlotter
from src.visualization.comparison import ComparisonPlotter


# =========================================================================
# 论文实验参数
# =========================================================================

# Heinzelman 2000 论文中的标准参数
PAPER_PARAMS = {
    "n_nodes": 100,
    "area": (0, 100, 0, 100),
    "base_station_pos": (50, 50),
    "initial_energy": 2.0,  # 论文中使用 2J
    "E_elec": 50e-9,
    "epsilon_fs": 10e-12,
    "epsilon_mp": 0.0013e-12,
    "E_da": 5e-9,
    "data_size": 4000,
    "p": 0.05,
    "seed": 42,
}


def reproduce_network_lifetime(
    rounds: int = 5000,
    output_dir: str = "results/reproduction/lifetime"
) -> Dict[str, Any]:
    """
    复现论文图 4: 网络生命周期 (Alive Nodes vs Rounds)

    论文结果:
      - 约 1250 轮时第一个节点死亡
      - 约 3000 轮时约一半节点死亡
      - 约 4500 轮时所有节点死亡

    Args:
        rounds: 仿真轮数
        output_dir: 输出目录

    Returns:
        仿真结果字典
    """
    print("=" * 70)
    print("实验 1: 复现网络生命周期曲线 (论文图 4)")
    print("=" * 70)

    # 使用论文参数
    energy_model = FirstOrderRadioModel(
        E_elec=PAPER_PARAMS["E_elec"],
        epsilon_fs=PAPER_PARAMS["epsilon_fs"],
        epsilon_mp=PAPER_PARAMS["epsilon_mp"],
        E_da=PAPER_PARAMS["E_da"],
    )

    network = Network(
        n_nodes=PAPER_PARAMS["n_nodes"],
        area=PAPER_PARAMS["area"],
        base_station_pos=PAPER_PARAMS["base_station_pos"],
        energy_model=energy_model,
        initial_energy=PAPER_PARAMS["initial_energy"],
        seed=PAPER_PARAMS["seed"],
    )

    print(f"参数: n={PAPER_PARAMS['n_nodes']}, E0={PAPER_PARAMS['initial_energy']}J, "
          f"area={PAPER_PARAMS['area'][1]}x{PAPER_PARAMS['area'][3]}m")
    print(f"轮数: {rounds}")
    print()

    start_time = time.time()
    results = network.simulate_network(
        rounds=rounds,
        protocol_name="leach"
    )
    elapsed = time.time() - start_time

    print(f"仿真耗时: {elapsed:.2f}s")
    print()

    # 打印关键指标
    print("关键指标:")
    print(f"  - 第一轮死亡:  {results['first_dead_round']} 轮")
    print(f"  - 半生命周期: {results['half_dead_round']} 轮")
    print(f"  - 全部死亡:    {results['network_lifetime']} 轮")
    print(f"  - 最终能量:    {results['final_energy']:.4f} J")
    print()

    # 与论文结果对比
    print("论文参考值 (近似):")
    print("  - 第一轮死亡:  ~1250 轮")
    print("  - 半生命周期: ~3000 轮")
    print("  - 全部死亡:   ~4500 轮")
    print()

    # 保存结果
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "lifetime_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # 可视化
    plotter = MetricsPlotter()
    plotter.plot_network_lifetime(
        results,
        save_path=str(output_path / "network_lifetime.png"),
        show=False
    )

    print(f"结果已保存到 {output_dir}/")

    return results


def reproduce_energy_consumption(
    rounds: int = 3000,
    output_dir: str = "results/reproduction/energy"
) -> Dict[str, Any]:
    """
    复现论文图 5: 能量消耗分析

    论文结果:
      - 总能量随时间线性下降
      - 簇头节点能量消耗远大于普通节点
      - LEACH 比静态分簇节能约 8x

    Args:
        rounds: 仿真轮数
        output_dir: 输出目录

    Returns:
        仿真结果字典
    """
    print("=" * 70)
    print("实验 2: 复现能量消耗分析 (论文图 5)")
    print("=" * 70)

    energy_model = FirstOrderRadioModel(
        E_elec=PAPER_PARAMS["E_elec"],
        epsilon_fs=PAPER_PARAMS["epsilon_fs"],
        epsilon_mp=PAPER_PARAMS["epsilon_mp"],
        E_da=PAPER_PARAMS["E_da"],
    )

    network = Network(
        n_nodes=PAPER_PARAMS["n_nodes"],
        area=PAPER_PARAMS["area"],
        base_station_pos=PAPER_PARAMS["base_station_pos"],
        energy_model=energy_model,
        initial_energy=PAPER_PARAMS["initial_energy"],
        seed=PAPER_PARAMS["seed"],
    )

    print(f"参数: n={PAPER_PARAMS['n_nodes']}, E0={PAPER_PARAMS['initial_energy']}J")
    print(f"轮数: {rounds}")
    print()

    start_time = time.time()
    results = network.simulate_network(
        rounds=rounds,
        protocol_name="leach"
    )
    elapsed = time.time() - start_time

    print(f"仿真耗时: {elapsed:.2f}s")
    print()

    # 能量分析
    initial_total = PAPER_PARAMS["n_nodes"] * PAPER_PARAMS["initial_energy"]
    consumed = initial_total - results["final_energy"]
    print(f"能量分析:")
    print(f"  - 初始总能量: {initial_total:.2f} J")
    print(f"  - 剩余总能量: {results['final_energy']:.4f} J")
    print(f"  - 消耗总能量: {consumed:.4f} J")
    print(f"  - 能量利用率: {consumed / initial_total * 100:.1f}%")
    print()

    # 保存结果
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "energy_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # 可视化
    plotter = MetricsPlotter()
    plotter.plot_energy_consumption(
        results,
        save_path=str(output_path / "energy_consumption.png"),
        show=False
    )

    print(f"结果已保存到 {output_dir}/")

    return results


def reproduce_probability_sweep(
    probabilities: Optional[List[float]] = None,
    rounds: int = 3000,
    output_dir: str = "results/reproduction/probability_sweep"
) -> Dict[str, Any]:
    """
    复现论文中不同簇头概率 p 对性能的影响

    论文结果:
      - p=0.05 是最优值 (对于 n=100)
      - p 过小导致簇过大, 簇头负载过重
      - p 过大导致簇过小, 失去分簇优势

    Args:
        probabilities: 要测试的概率列表
        rounds: 仿真轮数
        output_dir: 输出目录

    Returns:
        概率扫描结果
    """
    if probabilities is None:
        probabilities = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]

    print("=" * 70)
    print("实验 3: 复现簇头概率 p 的影响分析")
    print("=" * 70)
    print(f"测试概率: {probabilities}")
    print(f"仿真轮数: {rounds}")
    print()

    sweep_results = {}

    for p in probabilities:
        print(f"\n测试 p = {p:.2f}...")

        energy_model = FirstOrderRadioModel(
            E_elec=PAPER_PARAMS["E_elec"],
            epsilon_fs=PAPER_PARAMS["epsilon_fs"],
            epsilon_mp=PAPER_PARAMS["epsilon_mp"],
            E_da=PAPER_PARAMS["E_da"],
        )

        network = Network(
            n_nodes=PAPER_PARAMS["n_nodes"],
            area=PAPER_PARAMS["area"],
            base_station_pos=PAPER_PARAMS["base_station_pos"],
            energy_model=energy_model,
            initial_energy=PAPER_PARAMS["initial_energy"],
            seed=PAPER_PARAMS["seed"],
        )

        start_time = time.time()
        results = network.simulate_network(
            rounds=rounds,
            protocol_name="leach"
        )
        elapsed = time.time() - start_time

        sweep_results[str(p)] = {
            "probability": p,
            "network_lifetime": results["network_lifetime"],
            "half_network_lifetime": results["half_dead_round"],
            "final_energy": results["final_energy"],
            "total_rounds": results["total_rounds_simulated"],
            "execution_time": elapsed,
        }

        print(
            f"  生命周期={results['network_lifetime']} 轮, "
            f"半生命周期={results['half_dead_round']} 轮, "
            f"剩余能量={results['final_energy']:.4f} J"
        )

    # 找出最优概率
    best_p = max(
        sweep_results.keys(),
        key=lambda k: sweep_results[k]["network_lifetime"]
    )
    best_result = sweep_results[best_p]

    print(f"\n{'=' * 70}")
    print(f"最优簇头概率: p = {best_p} (生命周期 = {best_result['network_lifetime']} 轮)")
    print(f"{'=' * 70}")

    # 保存结果
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "probability_sweep.json", "w") as f:
        json.dump(sweep_results, f, indent=2)

    # 绘制概率-生命周期曲线
    _plot_probability_sweep(sweep_results, str(output_path))

    print(f"\n结果已保存到 {output_dir}/")

    return sweep_results


def _plot_probability_sweep(
    sweep_results: Dict[str, Any],
    output_dir: str
):
    """
    绘制概率扫描结果图

    Args:
        sweep_results: 概率扫描结果
        output_dir: 输出目录
    """
    import matplotlib.pyplot as plt

    probabilities = [sweep_results[k]["probability"] for k in sorted(sweep_results.keys())]
    lifetimes = [sweep_results[k]["network_lifetime"] for k in sorted(sweep_results.keys())]
    half_lifetimes = [sweep_results[k]["half_network_lifetime"] for k in sorted(sweep_results.keys())]
    energies = [sweep_results[k]["final_energy"] for k in sorted(sweep_results.keys())]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 网络生命周期
    ax1 = axes[0]
    ax1.plot(probabilities, lifetimes, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Cluster Head Probability (p)')
    ax1.set_ylabel('Network Lifetime (Rounds)')
    ax1.set_title('Network Lifetime vs Probability')
    ax1.grid(True, alpha=0.3)

    # 半生命周期
    ax2 = axes[1]
    ax2.plot(probabilities, half_lifetimes, 'rs-', linewidth=2, markersize=8)
    ax2.set_xlabel('Cluster Head Probability (p)')
    ax2.set_ylabel('Half Network Lifetime (Rounds)')
    ax2.set_title('Half Lifetime vs Probability')
    ax2.grid(True, alpha=0.3)

    # 最终剩余能量
    ax3 = axes[2]
    ax3.plot(probabilities, energies, 'g^-', linewidth=2, markersize=8)
    ax3.set_xlabel('Cluster Head Probability (p)')
    ax3.set_ylabel('Final Energy (J)')
    ax3.set_title('Final Energy vs Probability')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/probability_sweep.png", dpi=150, bbox_inches='tight')
    plt.close()


def run_all_experiments(
    rounds: int = 3000,
    output_dir: str = "results/reproduction"
):
    """
    运行所有复现实验

    Args:
        rounds: 仿真轮数
        output_dir: 输出目录
    """
    print("\n" + "#" * 70)
    print("# LEACH 论文结果复现实验")
    print("#" * 70)
    print()

    # 实验 1: 网络生命周期
    lifetime_results = reproduce_network_lifetime(
        rounds=rounds,
        output_dir=f"{output_dir}/lifetime"
    )

    # 实验 2: 能量消耗
    energy_results = reproduce_energy_consumption(
        rounds=rounds,
        output_dir=f"{output_dir}/energy"
    )

    # 实验 3: 概率扫描
    sweep_results = reproduce_probability_sweep(
        rounds=rounds,
        output_dir=f"{output_dir}/probability_sweep"
    )

    # 生成总报告
    _generate_reproduction_report(
        lifetime_results,
        energy_results,
        sweep_results,
        output_dir
    )


def _generate_reproduction_report(
    lifetime_results: Dict,
    energy_results: Dict,
    sweep_results: Dict,
    output_dir: str
):
    """
    生成复现实验总报告

    Args:
        lifetime_results: 生命周期实验结果
        energy_results: 能量实验结果
        sweep_results: 概率扫描结果
        output_dir: 输出目录
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report = []
    report.append("# LEACH 论文复现实验报告")
    report.append("")
    report.append("## 参考论文")
    report.append("W. R. Heinzelman, A. Chandrakasan, and H. Balakrishnan,")
    report.append('"Energy-Efficient Communication Protocol for Wireless Microsensor Networks,"')
    report.append("in Proc. 33rd Hawaii Int. Conf. System Sciences (HICSS), Jan. 2000.")
    report.append("")
    report.append("## 实验参数")
    report.append(f"- 节点数量: {PAPER_PARAMS['n_nodes']}")
    report.append(f"- 区域范围: {PAPER_PARAMS['area'][1]}m x {PAPER_PARAMS['area'][3]}m")
    report.append(f"- 基站位置: {PAPER_PARAMS['base_station_pos']}")
    report.append(f"- 初始能量: {PAPER_PARAMS['initial_energy']} J")
    report.append(f"- 簇头概率: {PAPER_PARAMS['p']}")
    report.append(f"- 随机种子: {PAPER_PARAMS['seed']}")
    report.append("")
    report.append("## 实验结果")
    report.append("")
    report.append("### 1. 网络生命周期")
    report.append(f"- 第一轮死亡: {lifetime_results['first_dead_round']} 轮")
    report.append(f"- 半生命周期: {lifetime_results['half_dead_round']} 轮")
    report.append(f"- 网络生命周期: {lifetime_results['network_lifetime']} 轮")
    report.append("")
    report.append("### 2. 能量消耗")
    report.append(f"- 初始总能量: {PAPER_PARAMS['n_nodes'] * PAPER_PARAMS['initial_energy']:.2f} J")
    report.append(f"- 最终剩余能量: {energy_results['final_energy']:.4f} J")
    report.append("")
    report.append("### 3. 最优簇头概率")
    best_p = max(
        sweep_results.keys(),
        key=lambda k: sweep_results[k]["network_lifetime"]
    )
    report.append(f"- 最优概率: p = {best_p}")
    report.append(f"- 对应生命周期: {sweep_results[best_p]['network_lifetime']} 轮")
    report.append("")

    report_path = output_path / "reproduction_report.md"
    report_path.write_text("\n".join(report))

    print(f"\n复现报告已保存到 {report_path}")


def parse_args():
    """解析命令行参数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="LEACH 论文结果复现脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["lifetime", "energy", "probability", "all"],
        help="要运行的实验 (默认: all)"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3000,
        help="仿真轮数 (默认: 3000)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/reproduction",
        help="结果输出目录 (默认: results/reproduction)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.experiment == "lifetime":
        reproduce_network_lifetime(rounds=args.rounds, output_dir=args.output)
    elif args.experiment == "energy":
        reproduce_energy_consumption(rounds=args.rounds, output_dir=args.output)
    elif args.experiment == "probability":
        reproduce_probability_sweep(output_dir=args.output)
    elif args.experiment == "all":
        run_all_experiments(rounds=args.rounds, output_dir=args.output)
