"""高级特征工程"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from ..models.network import Network
from ..models.node import Node


@dataclass
class FeatureStats:
    """特征统计信息"""
    mean_energy: float
    std_energy: float
    center_x: float
    center_y: float
    n_alive: int
    density: float
    energy_histogram: np.ndarray = field(default_factory=lambda: np.array([]))


class AdvancedFeatureExtractor:
    """高级特征提取器"""
    
    def __init__(self, network: Network, n_bins: int = 10):
        """
        初始化特征提取器
        
        Args:
            network: 网络对象
            n_bins: 直方图分箱数
        """
        self.network = network
        self.n_bins = n_bins
        self._stats: Optional[FeatureStats] = None
        self._energy_ranks: Optional[np.ndarray] = None
        self._neighbor_cache: Dict[Tuple[int, float], List[int]] = {}
    
    def compute_stats(self) -> FeatureStats:
        """计算全局统计信息"""
        alive = self.network.alive_nodes
        
        if not alive:
            return FeatureStats(
                mean_energy=0, std_energy=0,
                center_x=0, center_y=0,
                n_alive=0, density=0
            )
        
        energies = np.array([n.energy for n in alive])
        positions = np.array([(n.x, n.y) for n in alive])
        
        area = (self.network.area[1] - self.network.area[0]) * \
               (self.network.area[3] - self.network.area[2])
        
        hist, _ = np.histogram(energies, bins=self.n_bins)
        
        self._stats = FeatureStats(
            mean_energy=np.mean(energies),
            std_energy=np.std(energies),
            center_x=np.mean(positions[:, 0]),
            center_y=np.mean(positions[:, 1]),
            n_alive=len(alive),
            density=len(alive) / area if area > 0 else 0,
            energy_histogram=hist
        )
        
        return self._stats
    
    def get_stats(self) -> FeatureStats:
        """获取统计信息（惰性计算）"""
        if self._stats is None:
            return self.compute_stats()
        return self._stats
    
    def extract_features(self, node: Node) -> np.ndarray:
        """
        提取节点完整特征向量
        
        Args:
            node: 目标节点
            
        Returns:
            特征向量
        """
        features = {}
        
        stats = self.get_stats()
        bs_pos = self.network.base_station.position
        
        features['x'] = node.x
        features['y'] = node.y
        
        dist_to_center = np.sqrt(
            (node.x - stats.center_x)**2 + (node.y - stats.center_y)**2
        )
        features['dist_to_center'] = dist_to_center
        
        features['energy'] = node.energy
        features['energy_ratio'] = node.energy / node.initial_energy
        
        if stats.std_energy > 1e-9:
            features['energy_zscore'] = (node.energy - stats.mean_energy) / stats.std_energy
        else:
            features['energy_zscore'] = 0.0
        
        features['energy_rank'] = self._get_energy_rank(node)
        
        features['dist_to_bs'] = node.distance_to(bs_pos)
        features['dist_to_bs_normalized'] = node.distance_to(bs_pos) / 100
        
        features['neighbor_count_20'] = len(
            self._get_neighbors_cached(node, 20)
        )
        features['neighbor_count_40'] = len(
            self._get_neighbors_cached(node, 40)
        )
        features['neighbor_count_60'] = len(
            self._get_neighbors_cached(node, 60)
        )
        
        features['avg_neighbor_energy'] = self._get_avg_neighbor_energy(node, 30)
        features['max_neighbor_energy'] = self._get_max_neighbor_energy(node, 40)
        
        features['total_transmissions'] = node.transmissions
        features['total_receptions'] = node.receptions
        features['comm_load'] = node.transmissions + node.receptions
        
        features['is_near_bs'] = 1.0 if node.distance_to(bs_pos) < 30 else 0.0
        features['is_in_center'] = 1.0 if dist_to_center < 25 else 0.0
        
        feature_order = [
            'x', 'y', 'dist_to_center',
            'energy', 'energy_ratio', 'energy_zscore', 'energy_rank',
            'dist_to_bs', 'dist_to_bs_normalized',
            'neighbor_count_20', 'neighbor_count_40', 'neighbor_count_60',
            'avg_neighbor_energy', 'max_neighbor_energy',
            'total_transmissions', 'total_receptions', 'comm_load',
            'is_near_bs', 'is_in_center'
        ]
        
        return np.array([features[k] for k in feature_order])
    
    def extract_batch(self, nodes: List[Node]) -> np.ndarray:
        """
        批量提取特征
        
        Args:
            nodes: 节点列表
            
        Returns:
            特征矩阵 (n_nodes, n_features)
        """
        return np.array([self.extract_features(node) for node in nodes])
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        return [
            'x', 'y', 'dist_to_center',
            'energy', 'energy_ratio', 'energy_zscore', 'energy_rank',
            'dist_to_bs', 'dist_to_bs_normalized',
            'neighbor_count_20', 'neighbor_count_40', 'neighbor_count_60',
            'avg_neighbor_energy', 'max_neighbor_energy',
            'total_transmissions', 'total_receptions', 'comm_load',
            'is_near_bs', 'is_in_center'
        ]
    
    def _get_neighbors_cached(
        self,
        node: Node,
        radius: float
    ) -> List[Node]:
        """获取邻居（带缓存）"""
        cache_key = (node.id, radius)
        
        if cache_key not in self._neighbor_cache:
            self._neighbor_cache[cache_key] = self.network.get_neighbors(
                node, radius
            )
        
        return self._neighbor_cache[cache_key]
    
    def _get_energy_rank(self, node: Node) -> float:
        """能量排名（归一化到 0-1）"""
        if self._energy_ranks is None:
            self._compute_energy_ranks()
        
        return self._energy_ranks.get(node.id, 0.5)
    
    def _compute_energy_ranks(self) -> None:
        """计算所有节点的能量排名"""
        alive = self.network.alive_nodes
        
        if not alive:
            self._energy_ranks = {}
            return
        
        sorted_energies = sorted(
            [(n.id, n.energy) for n in alive],
            key=lambda x: x[1],
            reverse=True
        )
        
        n = len(sorted_energies)
        self._energy_ranks = {
            node_id: (n - rank) / n
            for rank, (node_id, _) in enumerate(sorted_energies)
        }
    
    def _get_avg_neighbor_energy(self, node: Node, radius: float) -> float:
        """邻居平均能量"""
        neighbors = self._get_neighbors_cached(node, radius)
        
        if not neighbors:
            return 0.0
        
        return np.mean([n.energy for n in neighbors])
    
    def _get_max_neighbor_energy(self, node: Node, radius: float) -> float:
        """邻居最大能量"""
        neighbors = self._get_neighbors_cached(node, radius)
        
        if not neighbors:
            return 0.0
        
        return max(n.energy for n in neighbors)
    
    def get_importance_weights(self) -> Dict[str, float]:
        """
        获取特征重要性权重（基于领域知识的启发式权重）
        
        Returns:
            特征名到权重的映射
        """
        return {
            'energy_ratio': 2.5,
            'energy_zscore': 2.0,
            'energy_rank': 2.0,
            'dist_to_bs_normalized': 1.5,
            'neighbor_count_30': 1.2,
            'avg_neighbor_energy': 1.0,
            'is_near_bs': 0.8,
            'comm_load': 0.5,
        }
    
    def apply_weights(
        self,
        features: np.ndarray,
        weights: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        应用特征权重
        
        Args:
            features: 特征矩阵
            weights: 权重字典
            
        Returns:
            加权后的特征矩阵
        """
        if weights is None:
            weights = self.get_importance_weights()
        
        feature_names = self.get_feature_names()
        weight_vector = np.ones(len(feature_names))
        
        for i, name in enumerate(feature_names):
            if name in weights:
                weight_vector[i] = weights[name]
        
        return features * weight_vector
    
    def normalize_features(
        self,
        features: np.ndarray,
        method: str = 'zscore'
    ) -> np.ndarray:
        """
        归一化特征
        
        Args:
            features: 特征矩阵
            method: 归一化方法 ('zscore', 'minmax', 'robust')
            
        Returns:
            归一化后的特征矩阵
        """
        if len(features) == 0:
            return features
        
        if method == 'zscore':
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0)
            std = np.where(std < 1e-9, 1.0, std)
            return (features - mean) / std
        
        elif method == 'minmax':
            min_vals = np.min(features, axis=0)
            max_vals = np.max(features, axis=0)
            range_vals = max_vals - min_vals
            range_vals = np.where(range_vals < 1e-9, 1.0, range_vals)
            return (features - min_vals) / range_vals
        
        elif method == 'robust':
            median = np.median(features, axis=0)
            q75 = np.percentile(features, 75, axis=0)
            q25 = np.percentile(features, 25, axis=0)
            iqr = q75 - q25
            iqr = np.where(iqr < 1e-9, 1.0, iqr)
            return (features - median) / iqr
        
        return features
    
    def clear_cache(self) -> None:
        """清除缓存"""
        self._neighbor_cache.clear()
        self._energy_ranks = None
        self._stats = None


class FeatureSelector:
    """特征选择器"""
    
    def __init__(self, extractor: AdvancedFeatureExtractor):
        self.extractor = extractor
    
    def select_by_variance(
        self,
        features: np.ndarray,
        threshold: float = 0.01
    ) -> Tuple[np.ndarray, List[int]]:
        """
        基于方差的特征选择
        
        Args:
            features: 特征矩阵
            threshold: 方差阈值
            
        Returns:
            筛选后的特征矩阵和保留的索引
        """
        variances = np.var(features, axis=0)
        selected_indices = [i for i, v in enumerate(variances) if v >= threshold]
        
        return features[:, selected_indices], selected_indices
    
    def select_by_correlation(
        self,
        features: np.ndarray,
        threshold: float = 0.95
    ) -> Tuple[np.ndarray, List[int]]:
        """
        基于相关性的特征选择
        
        Args:
            features: 特征矩阵
            threshold: 相关系数阈值
            
        Returns:
            筛选后的特征矩阵和保留的索引
        """
        n_features = features.shape[1]
        corr_matrix = np.corrcoef(features.T)
        
        selected = list(range(n_features))
        removed = set()
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if abs(corr_matrix[i, j]) > threshold:
                    if j not in removed:
                        removed.add(j)
        
        selected_indices = [i for i in selected if i not in removed]
        
        return features[:, selected_indices], selected_indices
    
    def select_top_k(
        self,
        features: np.ndarray,
        k: int,
        importance_scores: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[int]]:
        """
        选择 Top-K 最重要的特征
        
        Args:
            features: 特征矩阵
            k: 选择数量
            importance_scores: 重要性分数
            
        Returns:
            筛选后的特征矩阵和保留的索引
        """
        if importance_scores is None:
            importance_scores = np.var(features, axis=0)
        
        top_k_indices = np.argsort(importance_scores)[-k:]
        selected_indices = sorted(top_k_indices.tolist())
        
        return features[:, selected_indices], selected_indices
