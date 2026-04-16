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
    
    def save(self, path: Optional[str] = None):
        """保存模型"""
        save_path = Path(path or self.model_path)
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_model(save_path)
    
    def load(self, path: Optional[str] = None):
        """加载模型"""
        load_path = Path(path or self.model_path)
        if load_path.exists():
            self._load_model(load_path)
            self.is_trained = True
    
    @abstractmethod
    def _save_model(self, path: Path):
        """保存模型实现"""
        pass
    
    @abstractmethod
    def _load_model(self, path: Path):
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
        
        # 提取特征
        features = self._extract_features(network)
        
        # 预测
        scores = self.predict(features)
        
        # 选择得分最高的节点作为簇头
        cluster_id = 0
        cluster_heads = []
        selected_ids = set()
        
        # 按分数排序
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
        bs_pos = network.base_station.position
        features = []
        
        for node in network.alive_nodes:
            feat = node.get_features(bs_pos)
            
            # 添加邻居密度特征
            neighbors = network.get_neighbors(node, 30.0)
            feat = np.append(feat, len(neighbors))
            
            # 添加到最近基站的距离
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
