"""PyTorch 簇头选择器"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import pickle

from .selector import AIClusterSelector


class MLPClassifier(nn.Module):
    """多层感知机分类器"""
    
    def __init__(self, input_dim: int, hidden_dims: list = [64, 32]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PyTorchClusterSelector(AIClusterSelector):
    """基于 PyTorch 的簇头选择器"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        hidden_dims: list = [64, 32],
        learning_rate: float = 0.001,
        **kwargs
    ):
        """
        初始化
        
        Args:
            model_path: 模型保存路径
            hidden_dims: 隐藏层维度
            learning_rate: 学习率
            **kwargs: 其他参数
        """
        super().__init__(model_path)
        
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model: Optional[MLPClassifier] = None
        self.optimizer = None
        self.criterion = nn.BCELoss()
    
    def _ensure_model(self, input_dim: int):
        """确保模型已初始化"""
        if self.model is None:
            self.model = MLPClassifier(input_dim, self.hidden_dims).to(self.device)
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate
            )
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        预测簇头概率
        
        Args:
            features: 特征矩阵
            
        Returns:
            簇头概率
        """
        if not self.is_trained or self.model is None:
            return np.ones(len(features)) / len(features)
        
        self.model.eval()
        
        X = torch.tensor(features, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            probs = self.model(X).cpu().numpy().ravel()
        
        return probs
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32, **kwargs):
        """
        训练模型
        
        Args:
            X: 特征矩阵
            y: 标签
            epochs: 训练轮数
            batch_size: 批大小
        """
        self._ensure_model(X.shape[1])
        
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
        
        self.is_trained = True
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """预测概率（兼容接口）"""
        return self.predict(features)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性（基于梯度）"""
        if not self.is_trained or self.model is None:
            return {}
        
        # 使用随机输入计算梯度作为重要性近似
        self.model.eval()
        
        X = torch.randn(1, len(self.feature_names), requires_grad=True).to(self.device)
        output = self.model(X)
        output.backward()
        
        importances = torch.abs(X.grad).mean(dim=0).cpu().numpy()
        
        return {
            name: float(imp)
            for name, imp in zip(self.feature_names, importances)
        }
    
    def _save_model(self, path: Path):
        """保存模型"""
        checkpoint = {
            'model_state_dict': self.model.state_dict() if self.model else None,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'feature_names': self.feature_names,
            'hidden_dims': self.hidden_dims,
            'learning_rate': self.learning_rate,
            'is_trained': self.is_trained,
        }
        
        torch.save(checkpoint, path)
    
    def _load_model(self, path: Path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.hidden_dims = checkpoint['hidden_dims']
        self.learning_rate = checkpoint['learning_rate']
        self.feature_names = checkpoint['feature_names']
        
        self.model = MLPClassifier(len(self.feature_names), self.hidden_dims).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.is_trained = checkpoint['is_trained']
