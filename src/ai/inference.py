# AI模型推理模块
"""LEACH协议算法的AI模型推理模块，用于在实际场景中使用训练好的模型进行簇首选择。"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from .model import ClusterAIModel

def ai_select_heads(nodes: list, model: ClusterAIModel, scaler: StandardScaler, P: float = 0.05) -> tuple:
    """
    使用AI模型选择簇首节点。
    
    参数:
    nodes: 所有节点的列表
    model: 训练好的AI模型
    scaler: 数据标准化器
    P: 预期簇首比例
    
    返回:
    heads: 选取的簇首节点列表
    members: 非簇首的成员节点列表
    head_indices: 簇首节点的索引列表
    """
    # 特征提取和标准化
    features = np.array(nodes)
    features_scaled = scaler.transform(features)
    
    # 使用模型进行预测
    predictions = model.predict(features_scaled)
    
    # 将预测结果转换为概率值
    probabilities = predictions.flatten()
    
    # 根据预期簇首比例P选择簇首
    num_nodes = len(nodes)
    num_heads = max(1, int(num_nodes * P))  # 确保至少有1个簇首
    
    # 选择概率最高的num_heads个节点作为簇首
    head_indices = np.argsort(probabilities)[::-1][:num_heads]
    
    # 分离簇首和成员节点
    heads = [nodes[i] for i in head_indices]
    members = [nodes[i] for i in range(num_nodes) if i not in head_indices]
    
    return heads, members, head_indices

def load_trained_model(model_path: str) -> ClusterAIModel:
    """
    加载训练好的AI模型。
    
    参数:
    model_path: 模型文件路径
    
    返回:
    model: 加载的AI模型
    """
    # 创建模型实例（输入形状会从保存的模型中自动获取）
    # 这里我们使用一个临时的输入形状，实际加载时会被覆盖
    model = ClusterAIModel((2,))  # 假设节点有2个特征：x和y坐标
    
    # 加载模型权重
    model.load_model(model_path)
    
    return model
