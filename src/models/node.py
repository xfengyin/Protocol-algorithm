"""节点模型"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum


class NodeRole(Enum):
    """节点角色"""
    NORMAL = "normal"
    CLUSTER_HEAD = "cluster_head"
    DEAD = "dead"


@dataclass
class Node:
    """传感器节点"""
    
    id: int
    x: float
    y: float
    initial_energy: float = 0.5  # 焦耳 (J)
    energy: float = 0.5
    role: NodeRole = NodeRole.NORMAL
    round_dead: Optional[int] = None
    cluster_id: Optional[int] = None
    cluster_head: Optional[Node] = field(default=None, repr=False)
    
    # 历史记录
    energy_history: List[float] = field(default_factory=list)
    transmissions: int = 0
    receptions: int = 0
    
    def __post_init__(self):
        self.energy_history = [self.energy]
    
    @property
    def is_alive(self) -> bool:
        """是否存活"""
        return self.energy > 0 and self.role != NodeRole.DEAD
    
    @property
    def is_cluster_head(self) -> bool:
        """是否为簇头"""
        return self.role == NodeRole.CLUSTER_HEAD
    
    @property
    def is_normal(self) -> bool:
        """是否为普通节点"""
        return self.role == NodeRole.NORMAL
    
    def distance_to(self, other: Node | Tuple[float, float]) -> float:
        """计算到另一个节点或坐标的距离"""
        if isinstance(other, Node):
            ox, oy = other.x, other.y
        else:
            ox, oy = other
        
        return np.sqrt((self.x - ox) ** 2 + (self.y - oy) ** 2)
    
    def consume_energy(self, amount: float):
        """消耗能量"""
        self.energy = max(0, self.energy - amount)
        self.energy_history.append(self.energy)
        
        if self.energy <= 0 and self.role != NodeRole.DEAD:
            self.role = NodeRole.DEAD
    
    def become_cluster_head(self, cluster_id: int):
        """成为簇头"""
        self.role = NodeRole.CLUSTER_HEAD
        self.cluster_id = cluster_id
    
    def join_cluster(self, cluster_head: Node, cluster_id: int):
        """加入簇"""
        self.role = NodeRole.NORMAL
        self.cluster_head = cluster_head
        self.cluster_id = cluster_id
    
    def leave_cluster(self):
        """离开簇"""
        self.cluster_head = None
        self.cluster_id = None
    
    def reset_role(self):
        """重置角色"""
        self.role = NodeRole.NORMAL
        self.cluster_head = None
        self.cluster_id = None
    
    def get_features(self, base_station_pos: Tuple[float, float]) -> np.ndarray:
        """获取AI特征向量"""
        dist_to_bs = self.distance_to(base_station_pos)
        
        return np.array([
            self.x,
            self.y,
            self.energy,
            self.energy / self.initial_energy,  # 剩余能量比例
            dist_to_bs,
            self.transmissions,
            self.receptions,
        ])
    
    def __repr__(self) -> str:
        return f"Node(id={self.id}, pos=({self.x:.1f}, {self.y:.1f}), energy={self.energy:.4f}, role={self.role.value})"
