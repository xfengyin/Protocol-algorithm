"""LEACH-C 集中式协议"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, List, Dict

from .base import LEACHProtocol
from ..models.cluster_head import ClusterHead

if TYPE_CHECKING:
    from ..models.network import Network


class LEACHC(LEACHProtocol):
    """
    LEACH-C 集中式协议
    
    由基站集中优化簇头选择，基于所有节点的位置和能量信息。
    使用贪婪算法或模拟退火选择最优簇头集合。
    
    优势:
        - 全局视角选择
        - 能量感知
        - 更均匀的簇分布
    """
    
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
        
        # 计算每个节点到基站的距离
        bs_pos = network.base_station.position
        node_info = []
        
        for node in alive_nodes:
            dist_to_bs = node.distance_to(bs_pos)
            node_info.append({
                'node': node,
                'energy_ratio': node.energy / node.initial_energy,
                'dist_to_bs': dist_to_bs,
                'score': node.energy / node.initial_energy
            })
        
        # 按能量分数排序
        node_info.sort(key=lambda x: x['score'], reverse=True)
        
        # 选择能量较高的节点作为簇头
        cluster_id = 0
        cluster_heads = []
        selected_ids = set()
        
        for info in node_info:
            if cluster_id >= n_clusters:
                break
            
            node = info['node']
            if node.id not in selected_ids:
                node.become_cluster_head(cluster_id)
                
                ch = ClusterHead(
                    node=node,
                    cluster_id=cluster_id
                )
                cluster_heads.append(ch)
                selected_ids.add(node.id)
                cluster_id += 1
        
        self.current_round += 1
        
        return cluster_heads
