"""AI 簇头选择器基类"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from ..models.network import Network
from ..models.cluster_head import ClusterHead
from ..models.node import Node


class AIClusterSelector(ABC):
    """AI 簇头选择器基类"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        初始化
        
        Args:
            model_path: 模型保存路径
        """
        self.model = None
        self.model_path = model_path
        self.is_trained = False
        self.feature_names = [
            "x", "y", "energy", "energy_ratio",
            "dist_to_bs", "transmissions", "receptions"
        ]
    
    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        预测簇头
        
        Args:
            features: 特征矩阵 (n_samples, n_features)
            
        Returns:
            预测概率或分数
        """
        pass
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """
        训练模型
        
        Args:
            X: 特征矩阵
            y: 标签 (1=簇头, 0=非簇头)
            **kwargs: 训练参数
        """
        pass
    
    def save(self, path: Optional[str] = None) -> None:
        """保存模型"""
        save_path = Path(path or self.model_path)
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_model(save_path)
    
    def load(self, path: Optional[str] = None) -> None:
        """加载模型"""
        load_path = Path(path or self.model_path)
        if load_path.exists():
            self._load_model(load_path)
            self.is_trained = True
    
    @abstractmethod
    def _save_model(self, path: Path) -> None:
        """保存模型实现"""
        pass
    
    @abstractmethod
    def _load_model(self, path: Path) -> None:
        """加载模型实现"""
        pass
    
    def select_cluster_heads(
        self,
        network: Network,
        n_clusters: Optional[int] = None,
        **kwargs
    ) -> List[ClusterHead]:
        """
        基于 AI 模型选择簇头
        
        Args:
            network: 网络对象
            n_clusters: 目标簇数量
            
        Returns:
            簇头列表
        """
        alive_nodes = network.alive_nodes
        
        if n_clusters is None:
            n_clusters = max(1, int(len(alive_nodes) * 0.05))
        
        features = self._extract_features(network)
        
        scores = self.predict(features)
        
        cluster_id = 0
        cluster_heads = []
        selected_ids = set()
        
        sorted_indices = np.argsort(scores)[::-1]
        
        for idx in sorted_indices:
            if cluster_id >= n_clusters:
                break
            
            node = alive_nodes[idx]
            if node.id not in selected_ids:
                node.become_cluster_head(cluster_id)
                
                ch = ClusterHead(
                    node=node,
                    cluster_id=cluster_id
                )
                cluster_heads.append(ch)
                selected_ids.add(node.id)
                cluster_id += 1
        
        return cluster_heads
    
    def _extract_features(self, network: Network) -> np.ndarray:
        """
        提取节点特征
        
        Args:
            network: 网络对象
            
        Returns:
            特征矩阵
        """
        try:
            from ..ai.feature_engineering import AdvancedFeatureExtractor
            
            extractor = AdvancedFeatureExtractor(network)
            features = extractor.extract_batch(network.alive_nodes)
            return extractor.normalize_features(features, method='zscore')
        except ImportError:
            bs_pos = network.base_station.position
            features = []
            
            for node in network.alive_nodes:
                feat = node.get_features(bs_pos)
                
                neighbors = network.get_neighbors(node, 30.0)
                feat = np.append(feat, len(neighbors))
                feat = np.append(feat, node.distance_to(bs_pos))
                
                features.append(feat)
            
            return np.array(features) if features else np.array([]).reshape(0, len(self.feature_names) + 2)
    
    def explain_prediction(self, node: Node, network: Network) -> Dict[str, Any]:
        """
        解释为什么某个节点被选为簇头
        
        Args:
            node: 节点
            network: 网络对象
            
        Returns:
            解释字典
        """
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        features = node.get_features(network.base_station.position).reshape(1, -1)
        prediction = self.predict(features)[0]
        
        return {
            "node_id": node.id,
            "prediction_score": float(prediction),
            "features": {
                "energy_ratio": features[0][3],
                "dist_to_bs": features[0][4],
            }
        }
    
    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            X: 特征矩阵
            y_true: 真实标签
            threshold: 分类阈值
            
        Returns:
            评估指标字典
        """
        y_pred = (self.predict(X) >= threshold).astype(int)
        
        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
        }


class EnsembleClusterSelector:
    """集成簇头选择器"""
    
    def __init__(self, selectors: Optional[List[AIClusterSelector]] = None):
        """
        初始化集成选择器
        
        Args:
            selectors: 选择器列表
        """
        self.selectors = selectors or []
        self.weights = [1.0] * len(self.selectors)
    
    def add_selector(self, selector: AIClusterSelector, weight: float = 1.0) -> None:
        """
        添加选择器
        
        Args:
            selector: 选择器实例
            weight: 权重
        """
        self.selectors.append(selector)
        self.weights.append(weight)
    
    def set_weights(self, weights: List[float]) -> None:
        """
        设置权重
        
        Args:
            weights: 权重列表
        """
        if len(weights) != len(self.selectors):
            raise ValueError("Weights length must match selectors length")
        self.weights = weights
    
    def predict_scores(self, features: np.ndarray) -> np.ndarray:
        """
        获取加权平均分数
        
        Args:
            features: 特征矩阵
            
        Returns:
            加权分数
        """
        if not self.selectors:
            raise ValueError("No selectors available")
        
        scores = np.zeros(len(features))
        total_weight = sum(self.weights)
        
        for selector, weight in zip(self.selectors, self.weights):
            scores += weight * selector.predict(features)
        
        return scores / total_weight
    
    def select_cluster_heads(
        self,
        network: Network,
        n_clusters: Optional[int] = None,
        **kwargs
    ) -> List[ClusterHead]:
        """
        集成选择簇头
        
        Args:
            network: 网络对象
            n_clusters: 目标簇数量
            **kwargs: 其他参数
            
        Returns:
            簇头列表
        """
        alive_nodes = network.alive_nodes
        
        if n_clusters is None:
            n_clusters = max(1, int(len(alive_nodes) * 0.05))
        
        if not self.selectors:
            raise ValueError("No selectors configured")
        
        features = self._extract_features(network)
        scores = self.predict_scores(features)
        
        cluster_id = 0
        cluster_heads = []
        selected_ids = set()
        
        sorted_indices = np.argsort(scores)[::-1]
        
        for idx in sorted_indices:
            if cluster_id >= n_clusters:
                break
            
            node = alive_nodes[idx]
            if node.id not in selected_ids:
                node.become_cluster_head(cluster_id)
                
                ch = ClusterHead(
                    node=node,
                    cluster_id=cluster_id
                )
                cluster_heads.append(ch)
                selected_ids.add(node.id)
                cluster_id += 1
        
        return cluster_heads
    
    def _extract_features(self, network: Network) -> np.ndarray:
        """提取特征"""
        if self.selectors:
            return self.selectors[0]._extract_features(network)
        
        bs_pos = network.base_station.position
        features = []
        
        for node in network.alive_nodes:
            feat = node.get_features(bs_pos)
            features.append(feat)
        
        return np.array(features)
    
    def get_selector_info(self) -> List[Dict[str, Any]]:
        """
        获取选择器信息
        
        Returns:
            选择器信息列表
        """
        return [
            {
                "type": selector.__class__.__name__,
                "weight": weight,
                "trained": selector.is_trained,
            }
            for selector, weight in zip(self.selectors, self.weights)
        ]
