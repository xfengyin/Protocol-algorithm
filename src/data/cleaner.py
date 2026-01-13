# 数据清洗模块
"""用于清洗和预处理WSN节点数据，确保数据质量。"""

import numpy as np
from typing import List, Tuple

def clean_node_data(nodes: List[List[float]]) -> List[List[float]]:
    """
    清洗节点数据，移除无效数据。
    
    参数:
    nodes: 原始节点列表
    
    返回:
    cleaned_nodes: 清洗后的节点列表
    """
    cleaned_nodes = []
    
    for node in nodes:
        # 检查节点坐标是否有效
        if len(node) == 2:
            x, y = node
            # 检查坐标是否在合理范围内（0-1）
            if 0 <= x <= 1 and 0 <= y <= 1:
                cleaned_nodes.append([float(x), float(y)])
    
    return cleaned_nodes

def normalize_node_data(nodes: List[List[float]]) -> Tuple[List[List[float]], Tuple[float, float, float, float]]:
    """
    归一化节点坐标到0-1范围。
    
    参数:
    nodes: 原始节点列表
    
    返回:
    normalized_nodes: 归一化后的节点列表
    stats: 归一化统计信息，格式为(min_x, max_x, min_y, max_y)
    """
    if not nodes:
        return [], (0, 1, 0, 1)
    
    # 提取x和y坐标
    x_coords = [node[0] for node in nodes]
    y_coords = [node[1] for node in nodes]
    
    # 计算统计信息
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    # 避免除以零
    range_x = max_x - min_x if max_x != min_x else 1
    range_y = max_y - min_y if max_y != min_y else 1
    
    # 归一化
    normalized_nodes = [
        [(x - min_x) / range_x, (y - min_y) / range_y]
        for x, y in zip(x_coords, y_coords)
    ]
    
    return normalized_nodes, (min_x, max_x, min_y, max_y)

def remove_duplicate_nodes(nodes: List[List[float]], tolerance: float = 1e-6) -> List[List[float]]:
    """
    移除重复的节点。
    
    参数:
    nodes: 原始节点列表
    tolerance: 重复节点的距离阈值
    
    返回:
    unique_nodes: 去重后的节点列表
    """
    unique_nodes = []
    
    for new_node in nodes:
        is_duplicate = False
        for existing_node in unique_nodes:
            # 计算欧氏距离
            distance = np.sqrt(
                (new_node[0] - existing_node[0])**2 + 
                (new_node[1] - existing_node[1])**2
            )
            if distance < tolerance:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_nodes.append(new_node)
    
    return unique_nodes

def balance_dataset(nodes: List[List[float]], labels: List[int]) -> Tuple[List[List[float]], List[int]]:
    """
    平衡数据集，确保正负样本比例合理。
    
    参数:
    nodes: 节点列表
    labels: 标签列表
    
    返回:
    balanced_nodes: 平衡后的节点列表
    balanced_labels: 平衡后的标签列表
    """
    # 分离正负样本
    positive_nodes = [node for node, label in zip(nodes, labels) if label == 1]
    negative_nodes = [node for node, label in zip(nodes, labels) if label == 0]
    
    # 计算正负样本数量
    num_positive = len(positive_nodes)
    num_negative = len(negative_nodes)
    
    # 平衡样本数量（取较小值的2倍，或保持原比例）
    target_ratio = 0.5  # 目标正负样本比例
    
    if num_positive > num_negative * target_ratio:
        # 正样本过多，随机下采样
        sampled_positive = np.random.choice(len(positive_nodes), int(num_negative * target_ratio), replace=False)
        balanced_positive = [positive_nodes[i] for i in sampled_positive]
        balanced_negative = negative_nodes
    else:
        # 负样本过多，随机下采样
        sampled_negative = np.random.choice(len(negative_nodes), int(num_positive / target_ratio), replace=False)
        balanced_positive = positive_nodes
        balanced_negative = [negative_nodes[i] for i in sampled_negative]
    
    # 合并平衡后的样本
    balanced_nodes = balanced_positive + balanced_negative
    balanced_labels = [1] * len(balanced_positive) + [0] * len(balanced_negative)
    
    # 打乱顺序
    indices = np.arange(len(balanced_nodes))
    np.random.shuffle(indices)
    
    balanced_nodes = [balanced_nodes[i] for i in indices]
    balanced_labels = [balanced_labels[i] for i in indices]
    
    return balanced_nodes, balanced_labels
