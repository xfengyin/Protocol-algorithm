"""特征工程"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures


class FeatureEngineer:
    """特征工程"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
    
    def create_basic_features(
        self,
        x: np.ndarray,
        y: np.ndarray,
        energy: np.ndarray,
        dist_to_bs: np.ndarray
    ) -> Tuple[np.ndarray, List[str]]:
        """
        创建基础特征
        
        Args:
            x: x坐标
            y: y坐标
            energy: 剩余能量
            dist_to_bs: 到基站距离
            
        Returns:
            特征矩阵, 特征名列表
        """
        features = [
            x,
            y,
            energy,
            x ** 2,
            y ** 2,
            x * y,
            dist_to_bs,
            dist_to_bs ** 2,
        ]
        
        names = [
            "x", "y", "energy",
            "x_squared", "y_squared", "xy",
            "dist_to_bs", "dist_to_bs_squared"
        ]
        
        return np.column_stack(features), names
    
    def add_statistical_features(
        self,
        base_features: np.ndarray,
        names: List[str],
        x: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, List[str]]:
        """添加统计特征"""
        # 到其他节点的距离统计
        n = len(x)
        all_dists = np.sqrt(
            (x[:, np.newaxis] - x[np.newaxis, :]) ** 2 +
            (y[:, np.newaxis] - y[np.newaxis, :]) ** 2
        )
        
        # 移除对角线
        np.fill_diagonal(all_dists, np.inf)
        
        features = [
            np.min(all_dists, axis=1),  # 最近邻距离
            np.mean(all_dists, axis=1), # 平均距离
            np.std(all_dists, axis=1),  # 距离标准差
        ]
        
        new_names = ["min_neighbor_dist", "mean_neighbor_dist", "std_neighbor_dist"]
        
        return np.column_stack([base_features] + features), names + new_names
    
    def scale_features(
        self,
        features: np.ndarray,
        method: str = "standard"
    ) -> np.ndarray:
        """
        特征缩放
        
        Args:
            features: 特征矩阵
            method: 'standard' 或 'minmax'
            
        Returns:
            缩放后的特征
        """
        if method == "standard":
            return self.scaler.fit_transform(features)
        elif method == "minmax":
            scaler = MinMaxScaler()
            return scaler.fit_transform(features)
        else:
            return features
    
    def select_features(
        self,
        features: np.ndarray,
        names: List[str],
        importance: Dict[str, float],
        top_k: int = 5
    ) -> Tuple[np.ndarray, List[str]]:
        """
        特征选择
        
        Args:
            features: 特征矩阵
            names: 特征名列表
            importance: 特征重要性
            top_k: 选择前k个
            
        Returns:
            选中的特征, 选中的特征名
        """
        # 按重要性排序
        sorted_features = sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        selected_names = [name for name, _ in sorted_features]
        selected_indices = [names.index(name) for name in selected_names]
        
        return features[:, selected_indices], selected_names
    
    def create_polynomial_features(
        self,
        features: np.ndarray,
        degree: int = 2,
        interaction_only: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        创建多项式特征
        
        Args:
            features: 特征矩阵
            degree: 多项式阶数
            interaction_only: 仅交互项
            
        Returns:
            多项式特征, 特征名
        """
        poly = PolynomialFeatures(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=False
        )
        
        poly_features = poly.fit_transform(features)
        
        # 生成特征名
        n_original = features.shape[1]
        names = [f"poly_{i}" for i in range(poly_features.shape[1])]
        
        return poly_features, names
