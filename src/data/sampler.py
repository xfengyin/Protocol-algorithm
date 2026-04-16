"""不平衡采样器"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.utils import resample


class ImbalancedSampler:
    """不平衡数据采样器"""
    
    @staticmethod
    def undersample(
        X: np.ndarray,
        y: np.ndarray,
        ratio: float = 0.1,
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        欠采样
        
        Args:
            X: 特征
            y: 标签
            ratio: 正负例比例
            random_state: 随机种子
            
        Returns:
            采样后的 X, y
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # 分离正负例
        pos_mask = y == 1
        neg_mask = y == 0
        
        X_pos = X[pos_mask]
        X_neg = X[neg_mask]
        y_pos = y[pos_mask]
        y_neg = y[neg_mask]
        
        # 调整负例数量
        n_pos = len(X_pos)
        n_neg_desired = int(n_pos / ratio) - n_pos
        n_neg_desired = min(n_neg_desired, len(X_neg))
        
        if len(X_neg) > n_neg_desired:
            indices = np.random.choice(len(X_neg), n_neg_desired, replace=False)
            X_neg = X_neg[indices]
            y_neg = y_neg[indices]
        
        # 合并
        X_resampled = np.vstack([X_pos, X_neg])
        y_resampled = np.concatenate([y_pos, y_neg])
        
        # 打乱
        shuffle_idx = np.random.permutation(len(y_resampled))
        
        return X_resampled[shuffle_idx], y_resampled[shuffle_idx]
    
    @staticmethod
    def oversample(
        X: np.ndarray,
        y: np.ndarray,
        ratio: float = 0.1,
        method: str = "smote",
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        过采样
        
        Args:
            X: 特征
            y: 标签
            ratio: 正负例比例
            method: 'smote' 或 'random'
            random_state: 随机种子
            
        Returns:
            采样后的 X, y
        """
        pos_mask = y == 1
        neg_mask = y == 0
        
        X_pos = X[pos_mask]
        X_neg = X[neg_mask]
        y_pos = y[pos_mask]
        y_neg = y[neg_mask]
        
        n_neg = len(X_neg)
        n_pos_needed = int(n_neg * ratio)
        
        if method == "smote":
            X_pos_resampled = ImbalancedSampler._smote(X_pos, n_pos_needed, random_state)
        else:
            indices = np.random.choice(len(X_pos), n_pos_needed, replace=True)
            X_pos_resampled = X_pos[indices]
        
        y_pos_resampled = np.ones(len(X_pos_resampled))
        
        X_resampled = np.vstack([X_pos_resampled, X_neg])
        y_resampled = np.concatenate([y_pos_resampled, y_neg])
        
        shuffle_idx = np.random.RandomState(random_state).permutation(len(y_resampled))
        
        return X_resampled[shuffle_idx], y_resampled[shuffle_idx]
    
    @staticmethod
    def _smote(
        X: np.ndarray,
        n_samples: int,
        random_state: Optional[int] = None
    ) -> np.ndarray:
        """
        SMOTE 过采样
        
        Args:
            X: 正例样本
            n_samples: 目标样本数
            random_state: 随机种子
            
        Returns:
            生成的样本
        """
        if len(X) >= n_samples:
            return X
        
        rng = np.random.RandomState(random_state)
        
        synthetic = []
        
        while len(synthetic) < n_samples - len(X):
            idx = rng.randint(0, len(X))
            
            # 找到最近邻
            dists = np.linalg.norm(X - X[idx], axis=1)
            dists[idx] = np.inf
            nn_idx = np.argmin(dists)
            
            # 插值
            diff = X[nn_idx] - X[idx]
            gap = rng.random()
            new_sample = X[idx] + gap * diff
            
            synthetic.append(new_sample)
        
        return np.vstack([X, np.array(synthetic)])
