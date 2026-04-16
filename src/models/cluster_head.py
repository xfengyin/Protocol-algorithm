"""簇头模型"""

from __future__ import annotations

import numpy as np
from typing import List, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .node import Node


@dataclass
class ClusterHead:
    """簇头"""
    
    node: "Node"
    cluster_id: int
    member_nodes: List["Node"] = field(default_factory=list)
    
    @property
    def n_members(self) -> int:
        """成员数量"""
        return len(self.member_nodes)
    
    @property
    def total_member_energy(self) -> float:
        """成员总能量"""
        return sum(n.energy for n in self.member_nodes)
    
    @property
    def average_member_energy(self) -> float:
        """成员平均能量"""
        if self.n_members == 0:
            return 0
        return self.total_member_energy / self.n_members
    
    def add_member(self, node: "Node"):
        """添加成员"""
        if node not in self.member_nodes:
            self.member_nodes.append(node)
    
    def remove_member(self, node: "Node"):
        """移除成员"""
        if node in self.member_nodes:
            self.member_nodes.remove(node)
    
    def clear_members(self):
        """清空成员"""
        self.member_nodes.clear()
    
    def __repr__(self) -> str:
        return f"ClusterHead(id={self.node.id}, members={self.n_members})"
