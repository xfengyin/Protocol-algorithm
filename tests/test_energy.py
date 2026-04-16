"""能量模型测试"""

import pytest
import numpy as np

from src.energy.radio_model import FirstOrderRadioModel


class TestFirstOrderRadioModel:
    """First Order Radio Model 测试"""
    
    @pytest.fixture
    def model(self):
        return FirstOrderRadioModel()
    
    def test_transmit_energy_free_space(self, model):
        """测试自由空间传输能耗"""
        distance = 50  # < d_threshold
        message_size = 4000  # bits
        
        energy = model.calc_transmit_energy(distance, message_size)
        
        # 验证能耗计算
        expected = (model.E_elec + model.epsilon_fs * distance**2) * message_size
        assert np.isclose(energy, expected)
        assert energy > 0
    
    def test_transmit_energy_multi_path(self, model):
        """测试多径衰落传输能耗"""
        distance = 100  # > d_threshold
        message_size = 4000
        
        energy = model.calc_transmit_energy(distance, message_size)
        
        expected = (model.E_elec + model.epsilon_mp * distance**4) * message_size
        assert np.isclose(energy, expected)
    
    def test_receive_energy(self, model):
        """测试接收能耗"""
        message_size = 4000
        
        energy = model.calc_receive_energy(message_size)
        
        expected = model.E_elec * message_size
        assert np.isclose(energy, expected)
    
    def test_aggregation_energy(self, model):
        """测试数据聚合能耗"""
        message_size = 4000
        
        energy = model.calc_aggregation_energy(message_size)
        
        expected = model.E_da * message_size
        assert np.isclose(energy, expected)
    
    def test_total_communication_energy(self, model):
        """测试总通信能耗"""
        tx_distance = 50
        rx_distance = 0
        message_size = 4000
        
        energy = model.calc_total_communication_energy(
            tx_distance, rx_distance, message_size
        )
        
        tx_energy = model.calc_transmit_energy(tx_distance, message_size)
        rx_energy = model.calc_receive_energy(message_size)
        
        assert np.isclose(energy, tx_energy + rx_energy)
    
    def test_transmission_mode(self, model):
        """测试传输模式判断"""
        # 自由空间
        assert model.get_transmission_mode(50) == 'free_space'
        
        # 多径衰落
        assert model.get_transmission_mode(100) == 'multi_path'
    
    def test_energy_budget_estimation(self, model):
        """测试能量预算估算"""
        n_nodes = 100
        n_rounds = 100
        avg_cluster_size = 19  # 100/5
        message_size = 4000
        
        budget = model.estimate_network_energy_budget(
            n_nodes, n_rounds, avg_cluster_size, message_size
        )
        
        assert budget > 0
        assert np.isfinite(budget)
