"""经典 LEACH 协议"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, List

from .base import LEACHProtocol
from ..models.cluster_head import ClusterHead

if TYPE_CHECKING:
    from ..models.network import Network
    from ..models.node import Node


class ClassicLEACH(LEACHProtocol):
    """
    经典 LEACH 协议
    
    轮式簇头选择，每个节点根据阈值函数决定是否成为簇头。
    
    阈值公式:
        T(n) = p / (1 - p * (r mod 1/p))  如果 n ∈ G
        T(n) = 0                           否则
    
    其中:
        p = 簇头比例
        r = 当前轮数
        G = 过去 1/p 轮中未成为簇头的节点集合
    """
    
    def select_cluster_heads(self, network: "Network", **kwargs) -> List[ClusterHead]:
        """
        选择簇头
        
        Args:
            network: 网络对象
            
        Returns:
            簇头列表
        """
        n_nodes = len(network.alive_nodes)
        n_clusters = max(1, int(n_nodes * self.p))
        
        # 分配簇ID
        cluster_id = 0
        cluster_heads = []
        
        # 打乱顺序以保证公平性
        nodes = network.alive_nodes.copy()
        np.random.shuffle(nodes)
        
        for node in nodes:
            if cluster_id >= n_clusters:
                break
            
            # 计算阈值
            threshold = self.get_threshold(self.current_round)
            
            # 生成随机数
            r = np.random.random()
            
            if r < threshold:
                # 成为簇头
                node.become_cluster_head(cluster_id)
                
                ch = ClusterHead(
                    node=node,
                    cluster_id=cluster_id
                )
                cluster_heads.append(ch)
                cluster_id += 1
        
        self.current_round += 1
        
        return cluster_heads
