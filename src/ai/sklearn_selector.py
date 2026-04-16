"""sklearn 簇头选择器"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import pickle

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from .selector import AIClusterSelector


class SklearnClusterSelector(AIClusterSelector):
    """基于 sklearn 的簇头选择器"""
    
    def __init__(
        self,
        model_type: str = "rf",
        model_path: Optional[str] = None,
        n_estimators: int = 100,
        **kwargs
    ):
        """
        初始化
        
        Args:
            model_type: 模型类型 ('rf' 或 'gb')
            model_path: 模型保存路径
            n_estimators: 决策树数量
            **kwargs: sklearn 参数
        """
        super().__init__(model_path)
        
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.scaler = StandardScaler()
        
        # 创建模型
        if model_type == "rf":
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=42,
                class_weight='balanced',  # 处理不平衡
                **kwargs
            )
        elif model_type == "gb":
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                random_state=42,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        预测簇头概率
        
        Args:
            features: 特征矩阵
            
        Returns:
            簇头概率
        """
        if not self.is_trained:
            # 未训练时返回均匀分布
            return np.ones(len(features)) / len(features)
        
        # 标准化
        features_scaled = self.scaler.transform(features)
        
        # 预测概率
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(features_scaled)
            if probs.shape[1] == 2:
                return probs[:, 1]  # 正类概率
            return probs.ravel()
        else:
            return self.model.predict(features_scaled).astype(float)
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """
        训练模型
        
        Args:
            X: 特征矩阵
            y: 标签
            **kwargs: 训练参数
        """
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练
        self.model.fit(X_scaled, y, **kwargs)
        self.is_trained = True
    
    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return {}
        
        importances = self.model.feature_importances_
        return {
            name: float(imp)
            for name, imp in zip(self.feature_names, importances)
        }
    
    def _save_model(self, path: Path):
        """保存模型"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'n_estimators': self.n_estimators,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def _load_model(self, path: Path):
        """加载模型"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.n_estimators = model_data['n_estimators']
