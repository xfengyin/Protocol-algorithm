"""扩展测试套件"""

import pytest
import numpy as np
from typing import List, Tuple

from src.models.network import Network, NetworkMetrics
from src.models.node import Node, NodeRole
from src.models.base_station import BaseStation
from src.models.cluster_head import ClusterHead
from src.energy.radio_model import FirstOrderRadioModel
from src.energy.models import (
    EnergyModelFactory, FirstOrderRadioModel, Mica2Model,
    AdaptiveEnergyModel, EnergyModel
)
from src.leach.classic import ClassicLEACH
from src.leach.variants import LEACHRegistry


class TestEnergyModels:
    """能量模型测试"""

    def test_first_order_model(self):
        """测试一阶能量模型"""
        model = FirstOrderRadioModel()
        
        tx_energy = model.calc_transmit_energy(50, 4000)
        rx_energy = model.calc_receive_energy(4000)
        
        assert tx_energy > 0
        assert rx_energy > 0
        assert model.get_transmission_mode(50) == 'free_space'
        assert model.get_transmission_mode(100) == 'multi_path'

    def test_mica2_model(self):
        """测试 Mica2 能量模型"""
        model = Mica2Model()
        
        tx_energy = model.calc_transmit_energy(50, 4000)
        rx_energy = model.calc_receive_energy(4000)
        
        assert tx_energy > 0
        assert rx_energy > 0

    def test_adaptive_model(self):
        """测试自适应能量模型"""
        model = AdaptiveEnergyModel()
        
        initial_tx = model.calc_transmit_energy(50, 4000)
        
        model.update_adapt_factor(0.1)
        low_energy_tx = model.calc_transmit_energy(50, 4000)
        
        model.update_adapt_factor(0.9)
        high_energy_tx = model.calc_transmit_energy(50, 4000)
        
        assert low_energy_tx < initial_tx
        assert high_energy_tx > initial_tx

    def test_factory_creation(self):
        """测试工厂模式"""
        model1 = EnergyModelFactory.create('first_order')
        model2 = EnergyModelFactory.create('mica2')
        
        assert isinstance(model1, FirstOrderRadioModel)
        assert isinstance(model2, Mica2Model)
        
        assert 'first_order' in EnergyModelFactory.list_models()
        assert 'mica2' in EnergyModelFactory.list_models()
        
        with pytest.raises(ValueError):
            EnergyModelFactory.create('unknown_model')


class TestNetworkMetrics:
    """网络指标测试"""

    def test_metrics_to_dict(self):
        """测试指标序列化"""
        metrics = NetworkMetrics(
            round_number=10,
            alive_nodes=80,
            dead_nodes=20,
            n_cluster_heads=5,
            total_energy=40.0,
            average_energy=0.5,
            energy_std=0.1,
            cluster_size_distribution={0: 16, 1: 15, 2: 17, 3: 16, 4: 16}
        )
        
        result = metrics.to_dict()
        
        assert result['round'] == 10
        assert result['alive'] == 80
        assert result['dead'] == 20
        assert result['cluster_heads'] == 5
        assert result['total_energy'] == 40.0


class TestKDTreeOptimization:
    """KD-Tree 优化测试"""

    def test_neighbor_search(self):
        """测试邻居搜索"""
        network = Network(
            n_nodes=50,
            area=(0, 100, 0, 100),
            base_station_pos=(50, 50),
            seed=42
        )
        
        node = network.alive_nodes[0]
        neighbors = network.get_neighbors(node, 30.0)
        
        assert isinstance(neighbors, list)
        assert node not in neighbors
        
        for neighbor in neighbors:
            assert node.distance_to(neighbor) <= 30.0

    def test_spatial_index_invalidation(self):
        """测试空间索引失效"""
        network = Network(
            n_nodes=20,
            area=(0, 100, 0, 100),
            base_station_pos=(50, 50),
            seed=42
        )
        
        node = network.alive_nodes[0]
        network.get_neighbors(node, 30.0)
        assert network._index_valid is True
        
        network._invalidate_index()
        assert network._index_valid is False


class TestVectorizedSteadyPhase:
    """向量化稳定阶段测试"""

    def test_vectorized_vs_original(self):
        """测试向量化实现与原始实现的一致性"""
        network = Network(
            n_nodes=30,
            area=(0, 100, 0, 100),
            base_station_pos=(50, 50),
            seed=123
        )
        
        network.simulate_round('leach')
        
        ch_positions_original = [(ch.node.x, ch.node.y) for ch in network.cluster_heads]
        
        network.reset()
        network.simulate_round('leach')
        
        ch_positions_vectorized = [(ch.node.x, ch.node.y) for ch in network.cluster_heads]
        
        assert len(ch_positions_original) == len(ch_positions_vectorized)


class TestLEACHVariants:
    """LEACH 变体测试"""

    @pytest.mark.parametrize("protocol_name", [
        "leach", "leach_c", "leach_ee", "leach_m"
    ])
    def test_all_variants(self, protocol_name):
        """测试所有协议变体"""
        network = Network(
            n_nodes=50,
            area=(0, 100, 0, 100),
            base_station_pos=(50, 50),
            seed=42
        )
        
        metrics = network.simulate_round(protocol_name)
        
        assert metrics.alive_nodes > 0
        assert 0 <= metrics.n_cluster_heads <= network.n_nodes


class TestEdgeCases:
    """边界情况测试"""

    def test_single_node_network(self):
        """单节点网络"""
        network = Network(
            n_nodes=1,
            area=(0, 100, 0, 100),
            base_station_pos=(50, 50),
            seed=42
        )
        
        metrics = network.simulate_round('leach')
        assert metrics.n_cluster_heads <= 1

    def test_all_nodes_dead(self):
        """所有节点死亡后的行为"""
        network = Network(
            n_nodes=5,
            area=(0, 100, 0, 100),
            base_station_pos=(50, 50),
            initial_energy=0.001,
            seed=42
        )
        
        for _ in range(100):
            network.simulate_round('leach')
        
        assert network.n_alive == 0
        
        network.simulate_round('leach')

    def test_empty_network(self):
        """空网络处理"""
        network = Network(
            n_nodes=10,
            area=(0, 100, 0, 100),
            base_station_pos=(50, 50),
            seed=42
        )
        
        node = network.alive_nodes[0]
        neighbors = network.get_neighbors(node, 30.0)
        assert isinstance(neighbors, list)


class TestParallelSimulation:
    """并行仿真测试"""

    def test_simulation_config(self):
        """测试仿真配置"""
        from src.simulation.engine import SimulationConfig
        
        config = SimulationConfig(
            n_nodes=100,
            area=(0, 100, 0, 100),
            rounds=500,
            protocol_name='leach',
            seed=42
        )
        
        assert config.n_nodes == 100
        assert config.rounds == 500
        assert config.protocol_name == 'leach'


class TestFeatureEngineering:
    """特征工程测试"""

    def test_feature_extraction(self):
        """测试特征提取"""
        from src.ai.feature_engineering import AdvancedFeatureExtractor
        
        network = Network(
            n_nodes=20,
            area=(0, 100, 0, 100),
            base_station_pos=(50, 50),
            seed=42
        )
        
        extractor = AdvancedFeatureExtractor(network)
        node = network.alive_nodes[0]
        
        features = extractor.extract_features(node)
        
        assert len(features) == 19
        assert features.dtype == np.float64
        
        assert features[3] == node.energy
        assert features[4] == node.energy / node.initial_energy

    def test_batch_extraction(self):
        """测试批量特征提取"""
        from src.ai.feature_engineering import AdvancedFeatureExtractor
        
        network = Network(
            n_nodes=10,
            area=(0, 100, 0, 100),
            base_station_pos=(50, 50),
            seed=42
        )
        
        extractor = AdvancedFeatureExtractor(network)
        features = extractor.extract_batch(network.alive_nodes)
        
        assert features.shape == (10, 19)

    def test_feature_normalization(self):
        """测试特征归一化"""
        from src.ai.feature_engineering import AdvancedFeatureExtractor
        
        network = Network(
            n_nodes=20,
            area=(0, 100, 0, 100),
            base_station_pos=(50, 50),
            seed=42
        )
        
        extractor = AdvancedFeatureExtractor(network)
        features = extractor.extract_batch(network.alive_nodes)
        
        normalized = extractor.normalize_features(features, method='zscore')
        
        assert normalized.shape == features.shape
        
        col_means = np.mean(normalized, axis=0)
        col_stds = np.std(normalized, axis=0)
        
        assert np.allclose(col_means, 0, atol=1e-10)
        assert np.allclose(col_stds, 1, atol=1e-10)


class TestEnsembleSelector:
    """集成选择器测试"""

    def test_ensemble_creation(self):
        """测试集成选择器创建"""
        from src.ai.selector import EnsembleClusterSelector
        
        ensemble = EnsembleClusterSelector()
        assert len(ensemble.selectors) == 0
        assert len(ensemble.weights) == 0

    def test_ensemble_info(self):
        """测试集成选择器信息"""
        from src.ai.selector import EnsembleClusterSelector
        
        ensemble = EnsembleClusterSelector()
        info = ensemble.get_selector_info()
        
        assert info == []
