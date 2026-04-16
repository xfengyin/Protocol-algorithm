"""网络模型"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

from .node import Node, NodeRole
from .base_station import BaseStation
from .cluster_head import ClusterHead
from ..energy.radio_model import FirstOrderRadioModel
from ..leach.variants import LEACHRegistry


@dataclass
class NetworkMetrics:
    """网络指标"""
    round_number: int
    alive_nodes: int
    dead_nodes: int
    n_cluster_heads: int
    total_energy: float
    average_energy: float
    energy_std: float
    cluster_size_distribution: Dict[int, int] = field(default_factory=dict)


class Network:
    """无线传感器网络"""
    
    def __init__(
        self,
        n_nodes: int,
        area: Tuple[float, float, float, float],
        base_station_pos: Tuple[float, float],
        energy_model: Optional[FirstOrderRadioModel] = None,
        initial_energy: float = 0.5,
        seed: Optional[int] = None,
    ):
        """
        初始化网络
        
        Args:
            n_nodes: 节点数量
            area: 区域范围 (x_min, x_max, y_min, y_max)
            base_station_pos: 基站位置
            energy_model: 能量模型
            initial_energy: 初始能量 (J)
            seed: 随机种子
        """
        self.n_nodes = n_nodes
        self.area = area
        self.initial_energy = initial_energy
        self.energy_model = energy_model or FirstOrderRadioModel()
        
        if seed is not None:
            np.random.seed(seed)
        
        # 创建基站
        self.base_station = BaseStation(*base_station_pos)
        
        # 创建节点
        self.nodes = self._initialize_nodes()
        
        # 簇头记录
        self.cluster_heads: List[ClusterHead] = []
        self.current_round = 0
        
        # 指标历史
        self.metrics_history: List[NetworkMetrics] = []
        
        # 算法注册表
        self._protocol_registry = LEACHRegistry()
    
    def _initialize_nodes(self) -> List[Node]:
        """初始化节点"""
        x_min, x_max, y_min, y_max = self.area
        
        nodes = []
        for i in range(self.n_nodes):
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            node = Node(
                id=i,
                x=x,
                y=y,
                initial_energy=self.initial_energy,
                energy=self.initial_energy
            )
            nodes.append(node)
        
        return nodes
    
    @property
    def alive_nodes(self) -> List[Node]:
        """存活节点"""
        return [n for n in self.nodes if n.is_alive]
    
    @property
    def dead_nodes(self) -> List[Node]:
        """死亡节点"""
        return [n for n in self.nodes if not n.is_alive]
    
    @property
    def n_alive(self) -> int:
        """存活节点数"""
        return len(self.alive_nodes)
    
    @property
    def total_energy(self) -> float:
        """网络总能量"""
        return sum(n.energy for n in self.nodes)
    
    def get_neighbors(self, node: Node, threshold_distance: float) -> List[Node]:
        """获取邻居节点"""
        return [
            n for n in self.alive_nodes
            if n.id != node.id and node.distance_to(n) <= threshold_distance
        ]
    
    def reset_nodes(self):
        """重置所有节点角色"""
        for node in self.nodes:
            node.reset_role()
    
    def setup_phase(self, protocol_name: str = "leach", **kwargs) -> List[ClusterHead]:
        """
        设置阶段：选举簇头并形成簇
        
        Args:
            protocol_name: 协议名称
            **kwargs: 协议参数
            
        Returns:
            选出的簇头列表
        """
        self.reset_nodes()
        
        # 获取协议
        protocol = self._protocol_registry.get(protocol_name)
        
        # 选举簇头
        cluster_heads = protocol.select_cluster_heads(self, **kwargs)
        
        # 记录簇头
        self.cluster_heads = cluster_heads
        self.base_station.record_round([ch.node.id for ch in cluster_heads])
        
        return cluster_heads
    
    def steady_phase(self, data_size: int = 4000):
        """
        稳定阶段：数据传输
        
        Args:
            data_size: 数据包大小 (bits)
        """
        for ch in self.cluster_heads:
            ch.clear_members()
        
        # 普通节点加入簇
        for node in self.alive_nodes:
            if not node.is_cluster_head:
                # 找到最近的簇头
                min_dist = float('inf')
                nearest_ch = None
                
                for ch in self.cluster_heads:
                    dist = node.distance_to(ch.node)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_ch = ch
                
                if nearest_ch:
                    node.join_cluster(nearest_ch.node, nearest_ch.cluster_id)
                    nearest_ch.add_member(node)
                    
                    # 能耗计算
                    tx_energy = self.energy_model.calc_transmit_energy(
                        min_dist, data_size
                    )
                    node.consume_energy(tx_energy)
        
        # 簇头聚合并发送给基站
        for ch in self.cluster_heads:
            if not ch.node.is_alive:
                continue
            
            # 接收能耗
            total_bits = data_size * ch.n_members
            rx_energy = self.energy_model.calc_receive_energy(total_bits)
            ch.node.consume_energy(rx_energy)
            
            # 聚合
            fusion_energy = self.energy_model.E_da * total_bits
            ch.node.consume_energy(fusion_energy)
            
            # 发送能耗
            dist_to_bs = ch.node.distance_to(self.base_station)
            tx_to_bs = self.energy_model.calc_transmit_energy(dist_to_bs, total_bits)
            ch.node.consume_energy(tx_to_bs)
    
    def simulate_round(self, protocol_name: str = "leach", **kwargs) -> NetworkMetrics:
        """
        模拟一轮
        
        Args:
            protocol_name: 协议名称
            **kwargs: 协议参数
            
        Returns:
            网络指标
        """
        # 设置阶段
        self.setup_phase(protocol_name, **kwargs)
        
        # 稳定阶段
        self.steady_phase()
        
        # 记录指标
        metrics = self._collect_metrics()
        self.metrics_history.append(metrics)
        
        self.current_round += 1
        
        return metrics
    
    def simulate_network(
        self,
        rounds: int,
        protocol_name: str = "leach",
        stop_condition: Optional[callable] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        模拟整个网络生命周期
        
        Args:
            rounds: 最大轮数
            protocol_name: 协议名称
            stop_condition: 停止条件函数
            **kwargs: 协议参数
            
        Returns:
            仿真结果字典
        """
        self.reset()
        
        results = {
            "rounds": [],
            "alive_nodes": [],
            "dead_nodes": [],
            "total_energy": [],
            "cluster_heads": [],
        }
        
        for r in range(rounds):
            metrics = self.simulate_round(protocol_name, **kwargs)
            
            results["rounds"].append(r)
            results["alive_nodes"].append(metrics.alive_nodes)
            results["dead_nodes"].append(metrics.dead_nodes)
            results["total_energy"].append(metrics.total_energy)
            results["cluster_heads"].append(metrics.n_cluster_heads)
            
            # 检查停止条件
            if stop_condition and stop_condition(self, r):
                break
        
        # 计算汇总指标
        first_dead_round = next(
            (i for i, n in enumerate(results["alive_nodes"]) if n < self.n_nodes),
            rounds
        )
        half_dead_round = next(
            (i for i, n in enumerate(results["alive_nodes"]) if n <= self.n_nodes / 2),
            rounds
        )
        
        results.update({
            "network_lifetime": first_dead_round,
            "half_network_lifetime": half_dead_round,
            "total_rounds_simulated": len(results["rounds"]),
            "final_energy": results["total_energy"][-1] if results["total_energy"] else 0,
            "first_dead_round": first_dead_round,
            "half_dead_round": half_dead_round,
        })
        
        return results
    
    def _collect_metrics(self) -> NetworkMetrics:
        """收集网络指标"""
        alive = self.alive_nodes
        energies = [n.energy for n in alive]
        
        # 簇大小分布
        cluster_sizes = {}
        for ch in self.cluster_heads:
            cluster_sizes[ch.cluster_id] = ch.n_members
        
        return NetworkMetrics(
            round_number=self.current_round,
            alive_nodes=len(alive),
            dead_nodes=len(self.dead_nodes),
            n_cluster_heads=len(self.cluster_heads),
            total_energy=sum(energies),
            average_energy=np.mean(energies) if energies else 0,
            energy_std=np.std(energies) if energies else 0,
            cluster_size_distribution=cluster_sizes
        )
    
    def reset(self):
        """重置网络"""
        for node in self.nodes:
            node.energy = node.initial_energy
            node.role = NodeRole.NORMAL
            node.round_dead = None
            node.energy_history = [node.initial_energy]
            node.transmissions = 0
            node.receptions = 0
        
        self.current_round = 0
        self.cluster_heads.clear()
        self.metrics_history.clear()
        self.base_station.reset()
    
    def get_node(self, node_id: int) -> Optional[Node]:
        """获取节点"""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    
    def __repr__(self) -> str:
        return f"Network(nodes={self.n_nodes}, alive={self.n_alive}, rounds={self.current_round})"
