# 数据生成模块
"""用于生成WSN节点数据和标注，为AI模型训练提供数据源。"""

import numpy as np
from typing import List, Tuple

def generate_wsn_data(node_counts: List[int] = [10, 50, 100, 200, 500]) -> List[Tuple[List[List[float]], List[int]]]:
    """
    生成不同规模的WSN节点数据和标注。
    
    参数:
    node_counts: 不同规模的节点数量列表
    
    返回:
    datasets: 数据集列表，每个元素是(node_list, label_list)元组
    """
    datasets = []
    
    for count in node_counts:
        # 生成节点坐标
        nodes = [[np.random.random(), np.random.random()] for _ in range(count)]
        
        # 使用LEACH算法的阈值生成标注
        # 这里使用简化的标注生成方法，实际应用中应使用完整的LEACH算法结果
        labels = generate_leach_labels(nodes)
        
        datasets.append((nodes, labels))
    
    return datasets

def generate_leach_labels(nodes: List[List[float]], P: float = 0.05) -> List[int]:
    """
    基于LEACH算法生成节点标注（是否为簇首）。
    
    参数:
    nodes: 节点列表
    P: 簇首选择概率
    
    返回:
    labels: 节点标注列表，1表示簇首，0表示成员
    """
    num_nodes = len(nodes)
    num_heads = max(1, int(num_nodes * P))  # 簇首数量
    
    # 随机选择num_heads个节点作为簇首
    labels = [0] * num_nodes
    head_indices = np.random.choice(num_nodes, num_heads, replace=False)
    
    for idx in head_indices:
        labels[idx] = 1
    
    return labels

def generate_diverse_distributions(node_count: int = 100, num_distributions: int = 5) -> List[List[List[float]]]:
    """
    生成不同分布的节点数据，用于测试模型的泛化能力。
    
    参数:
    node_count: 每种分布的节点数量
    num_distributions: 分布类型数量
    
    返回:
    distributions: 不同分布的节点数据列表
    """
    distributions = []
    
    # 1. 均匀分布
    uniform_nodes = [[np.random.random(), np.random.random()] for _ in range(node_count)]
    distributions.append(uniform_nodes)
    
    # 2. 高斯分布（中心集中）
    gaussian_nodes = [[np.random.normal(0.5, 0.1), np.random.normal(0.5, 0.1)] for _ in range(node_count)]
    distributions.append(gaussian_nodes)
    
    # 3. 随机分布
    random_nodes = [[np.random.random(), np.random.random()] for _ in range(node_count)]
    distributions.append(random_nodes)
    
    # 4. 环形分布
    ring_nodes = []
    for _ in range(node_count):
        angle = np.random.random() * 2 * np.pi
        radius = 0.3 + np.random.random() * 0.2  # 环形半径在0.3-0.5之间
        x = 0.5 + radius * np.cos(angle)
        y = 0.5 + radius * np.sin(angle)
        ring_nodes.append([x, y])
    distributions.append(ring_nodes)
    
    # 5. 网格分布
    grid_size = int(np.sqrt(node_count))
    grid_nodes = []
    for i in range(grid_size):
        for j in range(grid_size):
            x = (i + 0.5) / grid_size + np.random.normal(0, 0.02)  # 添加少量噪声
            y = (j + 0.5) / grid_size + np.random.normal(0, 0.02)
            grid_nodes.append([x, y])
    distributions.append(grid_nodes)
    
    return distributions
