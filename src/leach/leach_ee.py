"""LEACH-EE 能量均衡协议"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, List, Dict

from .base import LEACHProtocol
from ..models.cluster_head import ClusterHead

if TYPE_CHECKING:
    from ..models.network import Network


class LEACHEE(LEACHProtocol):
    """
    LEACH-EE 能量均衡协议
    
    在 LEACH 基础上增加能量均衡机制:
    - 考虑节点剩余能量
    - 邻居节点密度感知
    - 动态调整簇头概率
    """
    
    def __init__(
        self,
        p: float = 0.05,
        energy_threshold: float = 0.3,
        density_weight: float = 0.3
    ):
        """
        初始化
        
        Args:
            p: 基础簇头概率
            energy_threshold: 能量阈值（低于此值不参与簇头选举）
            density_weight: 密度权重
        """
        super().__init__(p)
        self.energy_threshold = energy_threshold
        self.density_weight = density_weight
    
    def select_cluster_heads(self, network: "Network", **kwargs) -> List[ClusterHead]:
        """
        选择簇头
        
        Args:
            network: 网络对象
            
        Returns:
            簇头列表
        """
        alive_nodes = network.alive_nodes
        n_nodes = len(alive_nodes)
        n_clusters = max(1, int(n_nodes * self.p))
        
        # 计算邻居密度
        neighbor_threshold = 30.0  # 邻居距离阈值
        density_map = {}
        
        for node in alive_nodes:
            neighbors = network.get_neighbors(node, neighbor_threshold)
            density_map[node.id] = len(neighbors)
        
        # 计算调整后的阈值
        cluster_id = 0
        cluster_heads = []
        
        nodes = alive_nodes.copy()
        np.random.shuffle(nodes)
        
        for node in nodes:
            if cluster_id >= n_clusters:
                break
            
            # 检查能量阈值
            energy_ratio = node.energy / node.initial_energy
            if energy_ratio < self.energy_threshold:
                continue
            
            # 计算动态概率
            base_threshold = self.get_threshold(self.current_round)
            
            # 能量因子（剩余能量越高，概率越高）
            energy_factor = energy_ratio
            
            # 密度因子（邻居越少，越容易成为簇头）
            density = density_map.get(node.id, 0)
            density_factor = 1 - (density / max(density_map.values())) * self.density_weight
            
            # 综合阈值
            adjusted_threshold = base_threshold * energy_factor * density_factor
            
            r = np.random.random()
            
            if r < adjusted_threshold:
                node.become_cluster_head(cluster_id)
                
                ch = ClusterHead(
                    node=node,
                    cluster_id=cluster_id
                )
                cluster_heads.append(ch)
                cluster_id += 1
        
        self.current_round += 1
        
        return cluster_heads
