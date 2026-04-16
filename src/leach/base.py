"""LEACH 协议基类"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Dict, Any

if TYPE_CHECKING:
    from ..models.network import Network
    from ..models.cluster_head import ClusterHead


class LEACHProtocol(ABC):
    """LEACH 协议基类"""
    
    def __init__(self, p: float = 0.05):
        """
        初始化
        
        Args:
            p: 簇头选择概率
        """
        self.p = p
        self.current_round = 0
    
    @abstractmethod
    def select_cluster_heads(self, network: "Network", **kwargs) -> List["ClusterHead"]:
        """
        选择簇头
        
        Args:
            network: 网络对象
            **kwargs: 额外参数
            
        Returns:
            簇头列表
        """
        pass
    
    def get_threshold(self, n_iterations: int) -> float:
        """
        获取阈值 T(n)
        
        Args:
            n_iterations: 当前迭代次数
            
        Returns:
            阈值
        """
        if n_iterations % int(1 / self.p) == 0:
            return self.p
        return self.p * (1 - self.p * (n_iterations % int(1 / self.p)))
    
    def reset(self):
        """重置协议状态"""
        self.current_round = 0
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"
