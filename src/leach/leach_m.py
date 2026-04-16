"""LEACH-M 移动节点协议"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, List, Dict

from .base import LEACHProtocol
from ..models.cluster_head import ClusterHead

if TYPE_CHECKING:
    from ..models.network import Network


class LEACHM(LEACHProtocol):
    """
    LEACH-M 移动节点协议
    
    支持移动节点的 LEACH 变体:
    - 动态更新位置
    - 基于预测的路由
    - 移动感知的簇头选择
    """
    
    def __init__(
        self,
        p: float = 0.05,
        mobility_speed_range: tuple = (0, 1.0)
    ):
        """
        初始化
        
        Args:
            p: 簇头概率
            mobility_speed_range: 移动速度范围 (m/s)
        """
        super().__init__(p)
        self.speed_range = mobility_speed_range
        
        # 节点速度记录
        self.node_speeds: Dict[int, float] = {}
        self.previous_positions: Dict[int, tuple] = {}
    
    def update_positions(self, network: "Network"):
        """
        更新移动节点位置
        
        Args:
            network: 网络对象
        """
        for node in network.alive_nodes:
            # 保存上一轮位置
            self.previous_positions[node.id] = (node.x, node.y)
            
            # 随机速度
            speed = np.random.uniform(*self.speed_range)
            self.node_speeds[node.id] = speed
            
            # 随机方向
            angle = np.random.uniform(0, 2 * np.pi)
            
            # 更新位置
            dx = speed * np.cos(angle)
            dy = speed * np.sin(angle)
            
            x_min, x_max, y_min, y_max = network.area
            
            node.x = np.clip(node.x + dx, x_min, x_max)
            node.y = np.clip(node.y + dy, y_min, y_max)
    
    def predict_next_position(self, node: "Node") -> tuple:
        """
        预测节点下一位置
        
        Args:
            node: 节点
            
        Returns:
            预测位置 (x, y)
        """
        prev_pos = self.previous_positions.get(node.id, (node.x, node.y))
        speed = self.node_speeds.get(node.id, 0)
        
        angle = np.random.uniform(0, 2 * np.pi)
        dx = speed * np.cos(angle)
        dy = speed * np.sin(angle)
        
        return (prev_pos[0] + dx, prev_pos[1] + dy)
    
    def select_cluster_heads(self, network: "Network", **kwargs) -> List[ClusterHead]:
        """
        选择簇头
        
        Args:
            network: 网络对象
            
        Returns:
            簇头列表
        """
        # 更新移动节点位置
        self.update_positions(network)
        
        alive_nodes = network.alive_nodes
        n_nodes = len(alive_nodes)
        n_clusters = max(1, int(n_nodes * self.p))
        
        cluster_id = 0
        cluster_heads = []
        
        nodes = alive_nodes.copy()
        np.random.shuffle(nodes)
        
        for node in nodes:
            if cluster_id >= n_clusters:
                break
            
            # 移动惩罚：移动速度越快的节点越不适合作为簇头
            speed = self.node_speeds.get(node.id, 0)
            mobility_penalty = 1 - (speed / max(self.speed_range)) * 0.5
            
            threshold = self.get_threshold(self.current_round) * mobility_penalty
            
            r = np.random.random()
            
            if r < threshold:
                node.become_cluster_head(cluster_id)
                
                ch = ClusterHead(
                    node=node,
                    cluster_id=cluster_id
                )
                cluster_heads.append(ch)
                cluster_id += 1
        
        self.current_round += 1
        
        return cluster_heads
