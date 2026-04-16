"""网络模型测试"""

import pytest
import numpy as np

from src.models.network import Network
from src.models.node import Node, NodeRole
from src.models.base_station import BaseStation
from src.energy.radio_model import FirstOrderRadioModel
from src.leach.classic import ClassicLEACH


class TestNode:
    """节点测试"""
    
    def test_node_creation(self):
        """测试节点创建"""
        node = Node(id=0, x=10.0, y=20.0)
        
        assert node.id == 0
        assert node.x == 10.0
        assert node.y == 20.0
        assert node.energy == node.initial_energy
        assert node.is_alive
    
    def test_consume_energy(self):
        """测试能量消耗"""
        node = Node(id=0, x=0, y=0, initial_energy=1.0)
        initial_energy = node.energy
        
        node.consume_energy(0.3)
        
        assert np.isclose(node.energy, initial_energy - 0.3)
        assert node.is_alive
    
    def test_node_death(self):
        """测试节点死亡"""
        node = Node(id=0, x=0, y=0, initial_energy=1.0)
        
        node.consume_energy(1.1)  # 消耗超过所有能量
        
        assert node.energy == 0
        assert not node.is_alive
        assert node.role == NodeRole.DEAD
    
    def test_distance_calculation(self):
        """测试距离计算"""
        node1 = Node(id=0, x=0, y=0)
        node2 = Node(id=1, x=3, y=4)
        
        dist = node1.distance_to(node2)
        
        assert np.isclose(dist, 5.0)


class TestNetwork:
    """网络测试"""
    
    @pytest.fixture
    def network(self):
        return Network(
            n_nodes=20,
            area=(0, 100, 0, 100),
            base_station_pos=(50, 50),
            initial_energy=0.5,
            seed=42
        )
    
    def test_network_creation(self, network):
        """测试网络创建"""
        assert len(network.nodes) == 20
        assert network.n_alive == 20
        assert network.base_station is not None
    
    def test_simulation_round(self, network):
        """测试单轮仿真"""
        metrics = network.simulate_round('leach')
        
        assert metrics.alive_nodes == 20
        assert metrics.n_cluster_heads > 0
        assert metrics.total_energy > 0
    
    def test_network_lifetime(self, network):
        """测试网络生命周期"""
        results = network.simulate_network(rounds=100)
        
        assert 'network_lifetime' in results
        assert 'alive_nodes' in results
        assert len(results['alive_nodes']) == results['total_rounds_simulated']
    
    def test_neighbors(self, network):
        """测试邻居发现"""
        node = network.alive_nodes[0]
        neighbors = network.get_neighbors(node, threshold_distance=30.0)
        
        assert isinstance(neighbors, list)
        assert node not in neighbors
    
    def test_reset(self, network):
        """测试网络重置"""
        network.simulate_round('leach')
        network.reset()
        
        assert network.n_alive == 20
        assert network.current_round == 0


class TestClassicLEACH:
    """经典 LEACH 测试"""
    
    def test_cluster_head_selection(self):
        """测试簇头选择"""
        network = Network(
            n_nodes=50,
            area=(0, 100, 0, 100),
            base_station_pos=(50, 50),
            seed=42
        )
        
        protocol = ClassicLEACH(p=0.05)
        cluster_heads = protocol.select_cluster_heads(network)
        
        # 预期约5%的节点成为簇头
        expected_ch_count = int(50 * 0.05)
        assert len(cluster_heads) <= expected_ch_count + 2  # 允许一些误差
    
    def test_threshold_calculation(self):
        """测试阈值计算"""
        protocol = ClassicLEACH(p=0.05)
        
        # 第一轮
        t1 = protocol.get_threshold(0)
        assert np.isclose(t1, 0.05)
        
        # 非第一轮
        t2 = protocol.get_threshold(1)
        assert t2 > 0.05
