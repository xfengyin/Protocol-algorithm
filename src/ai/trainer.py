"""AI 训练管道"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from sklearn.model_selection import train_test_split

from .sklearn_selector import SklearnClusterSelector
from .pytorch_selector import PyTorchClusterSelector


class AITrainer:
    """AI 训练管道"""
    
    def __init__(self, model_type: str = "sklearn", **kwargs):
        """
        初始化
        
        Args:
            model_type: 模型类型 ('sklearn' 或 'pytorch')
            **kwargs: 模型参数
        """
        self.model_type = model_type
        
        if model_type == "sklearn":
            self.selector = SklearnClusterSelector(**kwargs)
        elif model_type == "pytorch":
            self.selector = PyTorchClusterSelector(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def load_dataset(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载数据集
        
        Args:
            path: 数据文件路径
            
        Returns:
            X, y
        """
        df = pd.read_csv(path)
        
        # 假设最后一列是标签
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        return X, y
    
    def prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        准备训练数据
        
        Args:
            X: 特征
            y: 标签
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y  # 分层采样
        )
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            X: 特征
            y: 标签
            epochs: 训练轮数（仅 PyTorch）
            **kwargs: 其他参数
            
        Returns:
            训练结果
        """
        # 处理不平衡数据
        if self.model_type == "sklearn":
            self.selector.train(X, y, **kwargs)
        else:
            self.selector.train(X, y, epochs=epochs, **kwargs)
        
        # 计算训练准确率
        predictions = self.selector.predict(X)
        accuracy = np.mean((predictions > 0.5).astype(int) == y)
        
        return {
            "accuracy": accuracy,
            "n_samples": len(y),
            "positive_ratio": np.mean(y),
        }
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            X: 特征
            y: 标签
            
        Returns:
            评估指标
        """
        predictions = self.selector.predict(X)
        pred_labels = (predictions > 0.5).astype(int)
        
        # 计算各种指标
        tp = np.sum((pred_labels == 1) & (y == 1))
        tn = np.sum((pred_labels == 0) & (y == 0))
        fp = np.sum((pred_labels == 1) & (y == 0))
        fn = np.sum((pred_labels == 0) & (y == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = np.mean(pred_labels == y)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    
    def save(self, path: str):
        """保存模型"""
        self.selector.save(path)
    
    def load(self, path: str):
        """加载模型"""
        self.selector.load(path)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        return self.selector.get_feature_importance()
