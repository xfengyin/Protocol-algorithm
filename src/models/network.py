"""网络模型"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Protocol, runtime_checkable
from dataclasses import dataclass, field
from scipy.spatial import cKDTree

from .node import Node, NodeRole
from .base_station import BaseStation
from .cluster_head import ClusterHead
from ..energy.radio_model import FirstOrderRadioModel
from ..leach.variants import LEACHRegistry


@runtime_checkable
class EnergyModel(Protocol):
    """能量模型协议"""
    
    def calc_transmit_energy(self, distance: float, message_size: int) -> float:
        ...
    
    def calc_receive_energy(self, message_size: int) -> float:
        ...
    
    @property
    def E_da(self) -> float:
        ...


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
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典用于序列化"""
        return {
            'round': self.round_number,
            'alive': self.alive_nodes,
            'dead': self.dead_nodes,
            'cluster_heads': self.n_cluster_heads,
            'total_energy': self.total_energy,
            'avg_energy': self.average_energy,
            'energy_std': self.energy_std,
            'cluster_distribution': self.cluster_size_distribution
        }


class Network:
    """无线传感器网络"""
    
    def __init__(
        self,
        n_nodes: int,
        area: Tuple[float, float, float, float],
        base_station_pos: Tuple[float, float],
        energy_model: Optional[EnergyModel] = None,
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
        
        self.base_station = BaseStation(*base_station_pos)
        self.nodes = self._initialize_nodes()
        
        self.cluster_heads: List[ClusterHead] = []
        self.current_round = 0
        self.metrics_history: List[NetworkMetrics] = []
        
        self._protocol_registry = LEACHRegistry()
        
        self._spatial_index: Optional[cKDTree] = None
        self._index_valid = False
        self._alive_node_indices: List[int] = []
    
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
    
    def _build_spatial_index(self) -> None:
        """构建空间索引"""
        if not self._index_valid and self.alive_nodes:
            positions = np.array([(n.x, n.y) for n in self.alive_nodes])
            self._spatial_index = cKDTree(positions)
            self._alive_node_indices = [i for i, n in enumerate(self.nodes) if n.is_alive]
            self._index_valid = True
    
    def _invalidate_index(self) -> None:
        """使空间索引失效"""
        self._index_valid = False
        self._spatial_index = None
        self._alive_node_indices = []
    
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
        """
        获取邻居节点（使用 KD-Tree 加速）
        
        Args:
            node: 目标节点
            threshold_distance: 距离阈值
            
        Returns:
            邻居节点列表
        """
        self._build_spatial_index()
        
        if self._spatial_index is None:
            return []
        
        idx_list = self._spatial_index.query_ball_point(
            (node.x, node.y), threshold_distance
        )
        
        return [
            self.alive_nodes[i] for i in idx_list
            if self.alive_nodes[i].id != node.id
        ]
    
    def get_neighbors_vectorized(
        self,
        threshold_distance: float
    ) -> Dict[int, List[int]]:
        """
        批量获取所有节点的邻居（向量化版本）
        
        Args:
            threshold_distance: 距离阈值
            
        Returns:
            节点索引到邻居索引列表的映射
        """
        self._build_spatial_index()
        
        if self._spatial_index is None or not self.alive_nodes:
            return {}
        
        positions = np.array([(n.x, n.y) for n in self.alive_nodes])
        distances = self._spatial_index.cKDTree__repr__(positions)
        
        result = {}
        for i, node in enumerate(self.alive_nodes):
            neighbor_indices = self._spatial_index.query_ball_point(
                positions[i], threshold_distance
            )
            result[node.id] = [self.alive_nodes[j].id for j in neighbor_indices if j != i]
        
        return result
    
    def reset_nodes(self) -> None:
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
        self._invalidate_index()
        
        protocol = self._protocol_registry.get(protocol_name)
        cluster_heads = protocol.select_cluster_heads(self, **kwargs)
        
        self.cluster_heads = cluster_heads
        self.base_station.record_round([ch.node.id for ch in cluster_heads])
        
        return cluster_heads
    
    def steady_phase(self, data_size: int = 4000) -> None:
        """
        稳定阶段：数据传输（向量化优化）
        
        Args:
            data_size: 数据包大小 (bits)
        """
        for ch in self.cluster_heads:
            ch.clear_members()
        
        if not self.cluster_heads or not self.alive_nodes:
            return
        
        ch_positions = np.array([(ch.node.x, ch.node.y) for ch in self.cluster_heads])
        ch_ids = np.array([ch.cluster_id for ch in self.cluster_heads])
        
        alive_positions = np.array([(n.x, n.y) for n in self.alive_nodes])
        alive_nodes_list = self.alive_nodes
        
        distances = np.linalg.norm(
            alive_positions[:, np.newaxis] - ch_positions[np.newaxis, :],
            axis=2
        )
        
        nearest_ch_indices = np.argmin(distances, axis=1)
        min_distances = distances[np.arange(len(distances)), nearest_ch_indices]
        
        ch_member_counts = {}
        
        for i, node in enumerate(alive_nodes_list):
            nearest_idx = nearest_ch_indices[i]
            
            if node.is_cluster_head:
                continue
            
            ch = self.cluster_heads[nearest_idx]
            node.join_cluster(ch.node, ch_ids[nearest_idx])
            ch.add_member(node)
            
            tx_energy = self.energy_model.calc_transmit_energy(
                min_distances[i], data_size
            )
            node.consume_energy(tx_energy)
            
            ch_id = ch_ids[nearest_idx]
            ch_member_counts[ch_id] = ch_member_counts.get(ch_id, 0) + 1
        
        for ch in self.cluster_heads:
            if not ch.node.is_alive:
                continue
            
            total_bits = data_size * ch.n_members
            
            rx_energy = self.energy_model.calc_receive_energy(total_bits)
            ch.node.consume_energy(rx_energy)
            
            fusion_energy = self.energy_model.E_da * total_bits
            ch.node.consume_energy(fusion_energy)
            
            dist_to_bs = ch.node.distance_to(self.base_station)
            tx_to_bs = self.energy_model.calc_transmit_energy(dist_to_bs, total_bits)
            ch.node.consume_energy(tx_to_bs)
    
    def steady_phase_original(self, data_size: int = 4000) -> None:
        """
        原始稳定阶段实现（保留用于对比）
        
        Args:
            data_size: 数据包大小 (bits)
        """
        for ch in self.cluster_heads:
            ch.clear_members()
        
        for node in self.alive_nodes:
            if not node.is_cluster_head:
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
                    
                    tx_energy = self.energy_model.calc_transmit_energy(
                        min_dist, data_size
                    )
                    node.consume_energy(tx_energy)
        
        for ch in self.cluster_heads:
            if not ch.node.is_alive:
                continue
            
            total_bits = data_size * ch.n_members
            rx_energy = self.energy_model.calc_receive_energy(total_bits)
            ch.node.consume_energy(rx_energy)
            
            fusion_energy = self.energy_model.E_da * total_bits
            ch.node.consume_energy(fusion_energy)
            
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
        self.setup_phase(protocol_name, **kwargs)
        self.steady_phase()
        
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
            
            if stop_condition and stop_condition(self, r):
                break
        
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
    
    def reset(self) -> None:
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
        self._invalidate_index()
    
    def get_node(self, node_id: int) -> Optional[Node]:
        """获取节点"""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    
    def __repr__(self) -> str:
        return f"Network(nodes={self.n_nodes}, alive={self.n_alive}, rounds={self.current_round})"
