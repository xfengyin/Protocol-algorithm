# AI模型训练模块
"""LEACH协议算法的AI模型训练模块，用于训练和优化簇首选择模型。"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .model import ClusterAIModel

def prepare_data(nodes: list, labels: list) -> tuple:
    """
    准备训练数据，包括特征提取和数据标准化。
    
    参数:
    nodes: 节点数据列表
    labels: 节点标签列表（0或1，表示是否为簇首）
    
    返回:
    X_train: 训练特征
    y_train: 训练标签
    X_val: 验证特征
    y_val: 验证标签
    scaler: 数据标准化器
    """
    # 特征提取：使用节点坐标作为初始特征
    # 后续可以扩展更多特征，如能量、距离等
    features = np.array(nodes)
    
    # 数据标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        features_scaled, 
        np.array(labels), 
        test_size=0.2, 
        random_state=42,
        stratify=labels  # 保持样本比例平衡
    )
    
    return X_train, y_train, X_val, y_val, scaler

def train_model(nodes: list, labels: list, model_path: str = None, model_type: str = 'random_forest') -> ClusterAIModel:
    """
    训练AI模型并保存。
    
    参数:
    nodes: 节点数据列表
    labels: 节点标签列表
    model_path: 模型保存路径（可选）
    model_type: 模型类型
    
    返回:
    model: 训练好的AI模型
    """
    # 准备数据
    X_train, y_train, X_val, y_val, scaler = prepare_data(nodes, labels)
    
    # 创建模型
    model = ClusterAIModel(model_type)
    
    # 训练模型
    history = model.train(X_train, y_train, X_val, y_val)
    
    # 保存模型
    if model_path:
        model.save_model(model_path)
    
    return model

def evaluate_model(model: ClusterAIModel, X_val: np.ndarray, y_val: np.ndarray) -> dict:
    """
    评估模型性能。
    
    参数:
    model: 训练好的AI模型
    X_val: 验证特征
    y_val: 验证标签
    
    返回:
    metrics: 模型评估指标
    """
    # 使用模型进行预测
    predictions = model.predict(X_val)
    
    # 将概率转换为二分类结果
    y_pred = (predictions > 0.5).astype(int)
    
    # 计算评估指标
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1_score': f1_score(y_val, y_pred)
    }
    
    return metrics
