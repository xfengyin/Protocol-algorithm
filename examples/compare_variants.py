#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""多协议对比实验脚本

对比不同 LEACH 变体协议的性能,包括:
- 经典 LEACH
- LEACH-C (集中式)
- LEACH-EE (能量均衡)
- LEACH-M (移动节点)

生成对比图表、统计报告和性能排名。

用法:
    python examples/compare_variants.py --output results/comparison/
    python examples/compare_variants.py --rounds 2000 --n-runs 3
"""

from __future__ import annotations

import sys
import argparse
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.network import Network
from src.energy.radio_model import FirstOrderRadioModel
from src.visualization.comparison import ComparisonPlotter
from src.visualization.metrics_plots import MetricsPlotter


# =========================================================================
# 协议配置
# =========================================================================

PROTOCOLS = {
    "leach": {
        "name": "LEACH (经典)",
        "description": "经典轮式簇头选择, 概率 p=0.05",
    },
    "leach-c": {
        "name": "LEACH-C (集中式)",
        "description": "基站集中优化簇头选择, 基于全局能量和位置信息",
    },
    "leach-ee": {
        "name": "LEACH-EE (能量均衡)",
        "description": "考虑节点剩余能量和邻居密度, 动态调整簇头概率",
    },
    "leach-m": {
        "name": "LEACH-M (移动节点)",
        "description": "支持移动节点的 LEACH 变体, 移动感知簇头选择",
    },
}


def run_protocol_comparison(
    n_nodes: int = 100,
    rounds: int = 1000,
    protocols: Optional[List[str]] = None,
    n_runs: int = 1,
    seed: int = 42,
    output_dir: str = "results/comparison"
) -> Dict[str, Any]:
    """
    运行多协议对比实验

    对每个协议在相同网络条件下运行仿真, 收集并对比性能指标。

    Args:
        n_nodes: 节点数量
        rounds: 仿真轮数
        protocols: 要对比的协议列表, None 表示全部
        n_runs: 每个协议的重复运行次数 (用于统计方差)
        seed: 基础随机种子
        output_dir: 结果输出目录

    Returns:
        对比结果字典, 包含:
            - protocol_results: 每个协议的仿真结果
            - summary: 汇总统计
    """
    protocols = protocols or list(PROTOCOLS.keys())

    print("=" * 70)
    print("多协议对比实验")
    print("=" * 70)
    print(f"节点数量: {n_nodes}")
    print(f"仿真轮数: {rounds}")
    print(f"重复次数: {n_runs}")
    print(f"协议列表: {', '.join(protocols)}")
    print()

    all_results: Dict[str, List[Dict]] = {p: [] for p in protocols}

    for protocol in protocols:
        info = PROTOCOLS[protocol]
        print(f"\n{'─' * 50}")
        print(f"协议: {info['name']}")
        print(f"描述: {info['description']}")
        print(f"{'─' * 50}")

        for run_id in range(n_runs):
            run_seed = seed + run_id
            print(f"\n  运行 {run_id + 1}/{n_runs} (seed={run_seed})...")

            start_time = time.time()

            # 创建网络
            energy_model = FirstOrderRadioModel()
            network = Network(
                n_nodes=n_nodes,
                area=(0, 100, 0, 100),
                base_station_pos=(50, 50),
                energy_model=energy_model,
                initial_energy=0.5,
                seed=run_seed
            )

            # 运行仿真
            results = network.simulate_network(
                rounds=rounds,
                protocol_name=protocol
            )

            elapsed = time.time() - start_time
            results["execution_time"] = elapsed

            all_results[protocol].append(results)

            print(
                f"    生命周期={results['network_lifetime']} 轮, "
                f"半生命周期={results['half_dead_round']} 轮, "
                f"耗时={elapsed:.1f}s"
            )

    # =========================================================================
    # 计算汇总统计
    # =========================================================================
    summary = _compute_summary(all_results, n_nodes)

    # =========================================================================
    # 生成图表
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("生成对比图表和报告")
    print(f"{'=' * 70}")

    _generate_comparison_plots(
        all_results,
        protocols,
        summary,
        output_dir
    )

    return {
        "protocol_results": all_results,
        "summary": summary,
    }


def _compute_summary(
    all_results: Dict[str, List[Dict]],
    n_nodes: int
) -> Dict[str, Dict[str, Any]]:
    """
    计算汇总统计

    Args:
        all_results: 每个协议的仿真结果列表
        n_nodes: 节点数量

    Returns:
        汇总统计字典
    """
    summary = {}

    for protocol, results_list in all_results.items():
        lifetimes = [r["network_lifetime"] for r in results_list]
        half_lifetimes = [r["half_dead_round"] for r in results_list]
        final_energies = [r["final_energy"] for r in results_list]
        exec_times = [r.get("execution_time", 0) for r in results_list]

        summary[protocol] = {
            "network_lifetime": {
                "mean": float(np.mean(lifetimes)),
                "std": float(np.std(lifetimes)),
                "min": float(np.min(lifetimes)),
                "max": float(np.max(lifetimes)),
            },
            "half_network_lifetime": {
                "mean": float(np.mean(half_lifetimes)),
                "std": float(np.std(half_lifetimes)),
            },
            "final_energy": {
                "mean": float(np.mean(final_energies)),
                "std": float(np.std(final_energies)),
            },
            "execution_time": {
                "mean": float(np.mean(exec_times)),
                "total": float(np.sum(exec_times)),
            },
            "n_runs": len(results_list),
        }

    return summary


def _generate_comparison_plots(
    all_results: Dict[str, List[Dict]],
    protocols: List[str],
    summary: Dict[str, Dict],
    output_dir: str
):
    """
    生成对比图表

    对每个协议, 选取第一次运行的结果进行对比可视化。

    Args:
        all_results: 所有仿真结果
        protocols: 协议列表
        summary: 汇总统计
        output_dir: 输出目录
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 选取每个协议的第一次运行结果进行对比
    results_list = []
    protocol_names = []

    for p in protocols:
        info = PROTOCOLS[p]
        results_list.append(all_results[p][0])
        protocol_names.append(info["name"])

    # 综合对比图
    plotter = ComparisonPlotter()
    plotter.compare_protocols(
        results_list=results_list,
        protocol_names=protocol_names,
        save_path=str(output_path / "protocol_comparison.png"),
        show=False
    )

    # 对比报告
    plotter.generate_comparison_report(
        results_list=results_list,
        protocol_names=protocol_names,
        output_dir=output_dir
    )

    # 绘制能量对比柱状图
    _plot_energy_bar_chart(summary, protocols, str(output_path))

    print(f"所有图表已保存到 {output_dir}/")


def _plot_energy_bar_chart(
    summary: Dict[str, Dict],
    protocols: List[str],
    output_dir: str
):
    """
    绘制能量和生命周期对比柱状图

    Args:
        summary: 汇总统计
        protocols: 协议列表
        output_dir: 输出目录
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    names = [PROTOCOLS[p]["name"] for p in protocols]

    # 网络生命周期
    ax1 = axes[0]
    means = [summary[p]["network_lifetime"]["mean"] for p in protocols]
    stds = [summary[p]["network_lifetime"]["std"] for p in protocols]

    ax1.bar(names, means, yerr=stds, color=colors[:len(protocols)],
            alpha=0.8, capsize=5)
    ax1.set_ylabel('Rounds')
    ax1.set_title('Network Lifetime (Mean ± Std)')
    ax1.tick_params(axis='x', rotation=30)
    ax1.grid(axis='y', alpha=0.3)

    # 最终剩余能量
    ax2 = axes[1]
    energy_means = [summary[p]["final_energy"]["mean"] for p in protocols]
    energy_stds = [summary[p]["final_energy"]["std"] for p in protocols]

    ax2.bar(names, energy_means, yerr=energy_stds,
            color=colors[:len(protocols)], alpha=0.8, capsize=5)
    ax2.set_ylabel('Energy (J)')
    ax2.set_title('Final Residual Energy (Mean ± Std)')
    ax2.tick_params(axis='x', rotation=30)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/energy_bar_chart.png", dpi=150, bbox_inches='tight')
    plt.close()


def print_summary_table(summary: Dict[str, Dict], protocols: List[str]):
    """
    打印汇总表格到终端

    Args:
        summary: 汇总统计
        protocols: 协议列表
    """
    print("\n" + "=" * 80)
    print("对比结果汇总")
    print("=" * 80)

    header = f"{'协议':<25} {'平均生命周期':>12} {'标准差':>10} {'半生命周期':>12} {'最终能量(J)':>12}"
    print(header)
    print("-" * 80)

    for p in protocols:
        s = summary[p]
        name = PROTOCOLS[p]["name"]
        print(
            f"{name:<25} "
            f"{s['network_lifetime']['mean']:>12.1f} "
            f"{s['network_lifetime']['std']:>10.1f} "
            f"{s['half_network_lifetime']['mean']:>12.1f} "
            f"{s['final_energy']['mean']:>12.4f}"
        )

    # 排名
    print("\n" + "-" * 80)
    print("排名 (按平均网络生命周期):")
    print("-" * 80)

    ranked = sorted(
        protocols,
        key=lambda p: summary[p]["network_lifetime"]["mean"],
        reverse=True
    )

    for rank, p in enumerate(ranked, 1):
        name = PROTOCOLS[p]["name"]
        lifetime = summary[p]["network_lifetime"]["mean"]
        print(f"  {rank}. {name}: {lifetime:.1f} 轮")


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="多协议对比实验脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认对比所有协议
  python examples/compare_variants.py

  # 指定输出目录和轮数
  python examples/compare_variants.py --output results/my_comparison/ --rounds 2000

  # 只对比特定协议, 重复运行 3 次
  python examples/compare_variants.py --protocols leach leach-c --n-runs 3
        """
    )

    parser.add_argument(
        "--n-nodes",
        type=int,
        default=100,
        help="节点数量 (默认: 100)"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1000,
        help="仿真轮数 (默认: 1000)"
    )
    parser.add_argument(
        "--protocols",
        nargs="+",
        default=None,
        help="要对比的协议列表 (默认: 全部)"
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=1,
        help="每个协议的重复运行次数 (默认: 1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/comparison",
        help="结果输出目录 (默认: results/comparison)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 运行对比实验
    comparison = run_protocol_comparison(
        n_nodes=args.n_nodes,
        rounds=args.rounds,
        protocols=args.protocols,
        n_runs=args.n_runs,
        seed=args.seed,
        output_dir=args.output
    )

    # 打印汇总表格
    protocols_used = args.protocols or list(PROTOCOLS.keys())
    print_summary_table(comparison["summary"], protocols_used)
