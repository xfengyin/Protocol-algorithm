#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AI 增强 LEACH 协议运行示例

演示如何使用 AI 优化的簇头选择器替代传统 LEACH 的随机选择:
- 使用经典 LEACH 生成训练数据
- 训练 sklearn 随机森林分类器
- 使用训练好的 AI 模型进行簇头选择
- 对比 AI-LEACH 与经典 LEACH 的性能

用法:
    python examples/run_ai_leach.py
"""

from __future__ import annotations

import sys
import time
import numpy as np
from pathlib import Path

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.network import Network
from src.models.node import Node
from src.energy.radio_model import FirstOrderRadioModel
from src.ai.trainer import AITrainer
from src.ai.sklearn_selector import SklearnClusterSelector
from src.visualization.metrics_plots import MetricsPlotter
from src.visualization.comparison import ComparisonPlotter


def generate_training_data(
    n_nodes: int = 100,
    training_rounds: int = 500,
    seed: int = 42,
    positive_ratio: float = 0.1
) -> tuple:
    """
    使用经典 LEACH 生成训练数据

    通过运行经典 LEACH 仿真,收集每一轮中成为簇头的节点特征,
    构造监督学习数据集。

    Args:
        n_nodes: 节点数量
        training_rounds: 用于生成数据的仿真轮数
        seed: 随机种子
        positive_ratio: 正样本(簇头)比例

    Returns:
        (X, y) - 特征矩阵和标签数组
    """
    print("\n" + "=" * 60)
    print("步骤 1: 生成训练数据")
    print("=" * 60)

    network = Network(
        n_nodes=n_nodes,
        area=(0, 100, 0, 100),
        base_station_pos=(50, 50),
        initial_energy=0.5,
        seed=seed
    )

    features_list = []
    labels_list = []

    bs_pos = network.base_station.position

    for r in range(training_rounds):
        # 重置节点角色
        network.reset_nodes()

        # 获取存活节点
        alive_nodes = network.alive_nodes
        if not alive_nodes:
            break

        # 经典 LEACH 选择簇头
        n_clusters = max(1, int(len(alive_nodes) * 0.05))
        cluster_head_ids = set()

        nodes = alive_nodes.copy()
        np.random.shuffle(nodes)
        cluster_id = 0

        for node in nodes:
            if cluster_id >= n_clusters:
                break
            threshold = 0.05  # 简化阈值
            r_val = np.random.random()
            if r_val < threshold:
                cluster_head_ids.add(node.id)
                cluster_id += 1

        # 提取每个节点的特征
        for node in alive_nodes:
            feat = node.get_features(bs_pos)

            # 添加邻居数量特征
            neighbors = network.get_neighbors(node, 30.0)
            feat = np.append(feat, len(neighbors))

            # 添加到基站距离
            feat = np.append(feat, node.distance_to(bs_pos))

            features_list.append(feat)

            # 标签: 1=簇头, 0=非簇头
            label = 1 if node.id in cluster_head_ids else 0
            labels_list.append(label)

    X = np.array(features_list)
    y = np.array(labels_list)

    print(f"  - 生成样本数: {len(y)}")
    print(f"  - 正样本(簇头): {np.sum(y)} ({np.mean(y) * 100:.1f}%)")
    print(f"  - 负样本(非簇头): {np.sum(1 - y)} ({np.mean(1 - y) * 100:.1f}%)")
    print(f"  - 特征维度: {X.shape[1]}")

    return X, y


def train_ai_model(X: np.ndarray, y: np.ndarray) -> AITrainer:
    """
    训练 AI 簇头选择器

    Args:
        X: 特征矩阵
        y: 标签数组

    Returns:
        训练好的 AITrainer 实例
    """
    print("\n" + "=" * 60)
    print("步骤 2: 训练 AI 模型")
    print("=" * 60)

    # 创建训练器 (使用随机森林)
    trainer = AITrainer(
        model_type="sklearn",
        model_type_rf="rf",
        n_estimators=100
    )

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        X, y,
        test_size=0.2,
        random_state=42
    )

    print(f"  - 训练集: {len(X_train)} 样本")
    print(f"  - 测试集: {len(X_test)} 样本")

    # 训练模型
    start_time = time.time()
    train_result = trainer.train(X_train, y_train)
    train_time = time.time() - start_time

    print(f"  - 训练时间: {train_time:.2f} 秒")
    print(f"  - 训练准确率: {train_result['accuracy'] * 100:.1f}%")

    # 评估模型
    eval_result = trainer.evaluate(X_test, y_test)
    print(f"  - 测试准确率: {eval_result['accuracy'] * 100:.1f}%")
    print(f"  - 精确率:     {eval_result['precision'] * 100:.1f}%")
    print(f"  - 召回率:     {eval_result['recall'] * 100:.1f}%")
    print(f"  - F1 分数:    {eval_result['f1'] * 100:.1f}%")

    # 保存模型
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(model_dir / "ai_selector.pkl")
    trainer.save(model_path)
    print(f"  - 模型已保存: {model_path}")

    return trainer


def run_ai_leach_simulation(
    trainer: AITrainer,
    rounds: int = 1000,
    seed: int = 42,
    output_dir: str = "results/ai_leach"
) -> dict:
    """
    使用训练好的 AI 模型运行 LEACH 仿真

    注意: 当前版本的 Network.simulate_network 使用协议注册表,
    这里演示如何使用 AI 选择器手动控制簇头选择过程。

    Args:
        trainer: 训练好的 AI 训练器
        rounds: 仿真轮数
        seed: 随机种子
        output_dir: 结果输出目录

    Returns:
        仿真结果字典
    """
    print("\n" + "=" * 60)
    print("步骤 3: AI-LEACH 仿真")
    print("=" * 60)

    network = Network(
        n_nodes=100,
        area=(0, 100, 0, 100),
        base_station_pos=(50, 50),
        initial_energy=0.5,
        seed=seed
    )

    selector = trainer.selector
    bs_pos = network.base_station.position

    results = {
        "rounds": [],
        "alive_nodes": [],
        "dead_nodes": [],
        "total_energy": [],
        "cluster_heads": [],
    }

    print(f"开始 AI-LEACH 仿真 ({rounds} 轮)...")

    for r in range(rounds):
        # 重置节点角色
        network.reset_nodes()

        alive_nodes = network.alive_nodes
        if not alive_nodes:
            break

        # 提取特征
        features = []
        for node in alive_nodes:
            feat = node.get_features(bs_pos)
            neighbors = network.get_neighbors(node, 30.0)
            feat = np.append(feat, len(neighbors))
            feat = np.append(feat, node.distance_to(bs_pos))
            features.append(feat)

        features = np.array(features)

        # AI 预测
        scores = selector.predict(features)

        # 选择得分最高的节点作为簇头
        n_clusters = max(1, int(len(alive_nodes) * 0.05))
        cluster_id = 0
        selected_ids = set()

        sorted_indices = np.argsort(scores)[::-1]
        for idx in sorted_indices:
            if cluster_id >= n_clusters:
                break
            node = alive_nodes[idx]
            if node.id not in selected_ids:
                node.become_cluster_head(cluster_id)
                selected_ids.add(node.id)
                cluster_id += 1

        # 稳态阶段: 数据传输
        network.steady_phase()

        # 收集指标
        alive_count = network.n_alive
        dead_count = len(network.dead_nodes)
        total_energy = network.total_energy

        results["rounds"].append(r)
        results["alive_nodes"].append(alive_count)
        results["dead_nodes"].append(dead_count)
        results["total_energy"].append(total_energy)
        results["cluster_heads"].append(len(selected_ids))

    # 计算生命周期指标
    n_nodes = 100
    alive = results["alive_nodes"]

    first_dead = next(
        (i for i, n in enumerate(alive) if n < n_nodes),
        len(alive)
    )
    half_dead = next(
        (i for i, n in enumerate(alive) if n <= n_nodes / 2),
        len(alive)
    )

    results.update({
        "network_lifetime": first_dead,
        "half_network_lifetime": half_dead,
        "first_dead_round": first_dead,
        "half_dead_round": half_dead,
        "total_rounds_simulated": len(results["rounds"]),
        "final_energy": results["total_energy"][-1] if results["total_energy"] else 0,
    })

    print(f"仿真完成!")
    print(f"  - 第一轮有节点死亡: {results['first_dead_round']} 轮")
    print(f"  - 一半节点死亡:     {results['half_dead_round']} 轮")
    print(f"  - 网络生命周期:     {results['network_lifetime']} 轮")
    print(f"  - 最终存活节点:     {alive[-1]}/{n_nodes}")

    # 可视化
    print("\n生成可视化图表...")
    plotter = MetricsPlotter()
    plotter.plot_all_metrics(
        results,
        output_dir=output_dir,
        show=False
    )
    print(f"图表已保存到 {output_dir}/")

    return results


def compare_ai_vs_classic(
    classic_results: dict,
    ai_results: dict,
    output_dir: str = "results/ai_comparison"
):
    """
    对比 AI-LEACH 与经典 LEACH

    Args:
        classic_results: 经典 LEACH 仿真结果
        ai_results: AI-LEACH 仿真结果
        output_dir: 输出目录
    """
    print("\n" + "=" * 60)
    print("步骤 4: 性能对比")
    print("=" * 60)

    plotter = ComparisonPlotter()
    plotter.compare_protocols(
        results_list=[classic_results, ai_results],
        protocol_names=["Classic LEACH", "AI-LEACH"],
        save_path=f"{output_dir}/ai_vs_classic.png",
        show=False
    )

    plotter.generate_comparison_report(
        results_list=[classic_results, ai_results],
        protocol_names=["Classic LEACH", "AI-LEACH"],
        output_dir=output_dir
    )

    # 打印对比摘要
    print("\n对比摘要:")
    print(f"{'指标':<20} {'Classic LEACH':>15} {'AI-LEACH':>15}")
    print("-" * 52)
    print(
        f"{'网络生命周期(轮)':<20} "
        f"{classic_results['network_lifetime']:>15} "
        f"{ai_results['network_lifetime']:>15}"
    )
    print(
        f"{'半生命周期(轮)':<20} "
        f"{classic_results['half_network_lifetime']:>15} "
        f"{ai_results['half_network_lifetime']:>15}"
    )
    print(
        f"{'最终剩余能量(J)':<20} "
        f"{classic_results['final_energy']:>15.4f} "
        f"{ai_results['final_energy']:>15.4f}"
    )


if __name__ == "__main__":
    # 1. 生成训练数据
    X, y = generate_training_data(
        n_nodes=100,
        training_rounds=500,
        seed=42
    )

    # 2. 训练 AI 模型
    trainer = train_ai_model(X, y)

    # 3. 运行 AI-LEACH 仿真
    ai_results = run_ai_leach_simulation(
        trainer,
        rounds=1000,
        seed=42
    )

    # 运行经典 LEACH 作为对比
    print("\n" + "=" * 60)
    print("补充: 运行经典 LEACH 作为对比基准")
    print("=" * 60)

    classic_network = Network(
        n_nodes=100,
        area=(0, 100, 0, 100),
        base_station_pos=(50, 50),
        initial_energy=0.5,
        seed=42
    )

    classic_results = classic_network.simulate_network(
        rounds=1000,
        protocol_name="leach"
    )

    print(f"经典 LEACH - 生命周期: {classic_results['network_lifetime']} 轮")

    # 4. 对比
    compare_ai_vs_classic(classic_results, ai_results)
