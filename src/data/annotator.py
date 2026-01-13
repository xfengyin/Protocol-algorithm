# 数据标注模块
"""用于生成WSN节点数据的标注，为AI模型训练提供高质量的监督数据。"""

import numpy as np
from typing import List, Tuple
from src.leach.utils import distance

def generate_annotations(nodes: List[List[float]], use_leach: bool = True) -> List[int]:
    """
    为节点数据生成标注，指示每个节点是否应该成为簇首。
    
    参数:
    nodes: 节点列表
    use_leach: 是否使用LEACH算法生成标注
    
    返回:
    annotations: 节点标注列表，1表示簇首，0表示成员
    """
    if use_leach:
        # 使用LEACH算法生成标注
        return generate_leach_annotations(nodes)
    else:
        # 使用基于距离的标注生成方法
        return generate_distance_based_annotations(nodes)

def generate_leach_annotations(nodes: List[List[float]], P: float = 0.05, r: int = 0) -> List[int]:
    """
    基于完整的LEACH算法生成节点标注。
    
    参数:
    nodes: 节点列表
    P: 簇首选择概率
    r: 当前轮数
    
    返回:
    annotations: 节点标注列表
    """
    num_nodes = len(nodes)
    annotations = [0] * num_nodes
    
    # 计算LEACH阈值
    Tn = P / (1 - P * (r % (1 / P)))
    
    # 为每个节点生成随机数并判断是否成为簇首
    for i in range(num_nodes):
        rand = np.random.random()
        if rand <= Tn:
            annotations[i] = 1
    
    return annotations

def generate_distance_based_annotations(nodes: List[List[float]], num_heads: int = None, P: float = 0.05) -> List[int]:
    """
    基于距离的簇首选择标注生成方法。
    选择距离其他节点较远的节点作为簇首，以实现更好的簇覆盖。
    
    参数:
    nodes: 节点列表
    num_heads: 簇首数量（可选）
    P: 簇首选择概率（当num_heads为None时使用）
    
    返回:
    annotations: 节点标注列表
    """
    num_nodes = len(nodes)
    
    if num_heads is None:
        num_heads = max(1, int(num_nodes * P))
    
    annotations = [0] * num_nodes
    
    # 计算每个节点到其他所有节点的距离之和
    distance_scores = []
    for i in range(num_nodes):
        total_distance = 0
        for j in range(num_nodes):
            if i != j:
                total_distance += distance(nodes[i], nodes[j])
        distance_scores.append((total_distance, i))
    
    # 选择距离之和最大的num_heads个节点作为簇首（距离其他节点最远）
    distance_scores.sort(reverse=True)  # 按距离之和降序排序
    
    for _, idx in distance_scores[:num_heads]:
        annotations[idx] = 1
    
    return annotations

def calculate_network_metrics(nodes: List[List[float]], annotations: List[int]) -> dict:
    """
    计算网络的性能指标，用于评估标注质量。
    
    参数:
    nodes: 节点列表
    annotations: 节点标注列表
    
    返回:
    metrics: 网络性能指标字典
    """
    num_nodes = len(nodes)
    num_heads = sum(annotations)
    
    # 提取簇首和成员节点
    heads = [nodes[i] for i, label in enumerate(annotations) if label == 1]
    members = [nodes[i] for i, label in enumerate(annotations) if label == 0]
    
    # 计算每个成员节点到最近簇首的距离
    min_distances = []
    for member in members:
        if heads:
            min_dist = min(distance(member, head) for head in heads)
            min_distances.append(min_dist)
    
    # 计算簇首间的距离
    head_distances = []
    for i in range(len(heads)):
        for j in range(i + 1, len(heads)):
            head_distances.append(distance(heads[i], heads[j]))
    
    # 生成指标
    metrics = {
        'num_nodes': num_nodes,
        'num_heads': num_heads,
        'head_ratio': num_heads / num_nodes if num_nodes > 0 else 0,
        'avg_min_distance': np.mean(min_distances) if min_distances else 0,
        'std_min_distance': np.std(min_distances) if min_distances else 0,
        'avg_head_distance': np.mean(head_distances) if head_distances else 0,
        'std_head_distance': np.std(head_distances) if head_distances else 0
    }
    
    return metrics

def generate_training_data(nodes: List[List[float]], annotations: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成用于AI模型训练的特征和标签数据。
    
    参数:
    nodes: 节点列表
    annotations: 节点标注列表
    
    返回:
    X: 训练特征（numpy数组）
    y: 训练标签（numpy数组）
    """
    # 提取基本特征：节点坐标
    X = np.array(nodes)
    
    # 添加额外特征：与其他节点的距离统计
    X_with_features = add_node_features(X)
    
    # 标签
    y = np.array(annotations).reshape(-1, 1)
    
    return X_with_features, y

def add_node_features(X: np.ndarray) -> np.ndarray:
    """
    为节点特征添加额外的统计信息。
    
    参数:
    X: 基本节点特征（坐标）
    
    返回:
    X_with_features: 添加了额外特征的节点特征
    """
    num_nodes = X.shape[0]
    additional_features = []
    
    for i in range(num_nodes):
        # 计算当前节点到其他所有节点的距离
        distances = np.sqrt(np.sum((X - X[i])**2, axis=1))
        
        # 计算距离统计特征
        avg_distance = np.mean(distances)
        min_distance = np.min(distances)
        max_distance = np.max(distances)
        std_distance = np.std(distances)
        
        # 添加到额外特征列表
        additional_features.append([avg_distance, min_distance, max_distance, std_distance])
    
    # 合并基本特征和额外特征
    additional_features = np.array(additional_features)
    X_with_features = np.concatenate([X, additional_features], axis=1)
    
    return X_with_features
