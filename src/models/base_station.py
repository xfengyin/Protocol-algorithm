"""基站模型"""

from __future__ import annotations

import numpy as np
from typing import Tuple, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .node import Node


class BaseStation:
    """基站"""
    
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.position = (x, y)
        self.total_received_data = 0
        self.cluster_heads_history: List[List[int]] = []
    
    def distance_to(self, node: "Node" | Tuple[float, float]) -> float:
        """计算到节点的距离"""
        if isinstance(node, tuple):
            ox, oy = node
        else:
            ox, oy = node.x, node.y
        
        return np.sqrt((self.x - ox) ** 2 + (self.y - oy) ** 2)
    
    def receive_data(self, size_bits: int):
        """接收数据"""
        self.total_received_data += size_bits
    
    def record_round(self, cluster_head_ids: List[int]):
        """记录一轮的簇头ID"""
        self.cluster_heads_history.append(cluster_head_ids.copy())
    
    def reset(self):
        """重置"""
        self.total_received_data = 0
        self.cluster_heads_history.clear()
    
    def __repr__(self) -> str:
        return f"BaseStation(pos=({self.x}, {self.y}))"
