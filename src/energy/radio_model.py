"""First Order Radio Model - 能量模型"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class FirstOrderRadioModel:
    """
    First Order Radio Model
    
    能量消耗模型，包含：
    - E_elec: 发射/接收电路能耗
    - epsilon_fs: 自由空间能耗系数
    - epsilon_mp: 多径衰落能耗系数
    - d_threshold: 距离阈值
    - E_da: 数据聚合能耗
    """
    
    # 电路能耗 (J/bit)
    E_elec: float = 50e-9  # 50 nJ/bit
    
    # 放大器能耗 (J/bit/m^2 或 J/bit/m^4)
    epsilon_fs: float = 10e-12  # 10 pJ/bit/m^2 (自由空间)
    epsilon_mp: float = 0.0013e-12  # 0.0013 pJ/bit/m^4 (多径)
    
    # 距离阈值 (m)
    d_threshold: float = 87.0  # sqrt(epsilon_fs / epsilon_mp)
    
    # 数据聚合能耗 (J/bit)
    E_da: float = 5e-9  # 5 nJ/bit
    
    def calc_transmit_energy(self, distance: float, message_size: int) -> float:
        """
        计算发送能耗
        
        Args:
            distance: 传输距离 (m)
            message_size: 消息大小 (bits)
            
        Returns:
            发送能耗 (J)
        """
        if distance <= self.d_threshold:
            # 自由空间模型
            energy = (self.E_elec + self.epsilon_fs * distance ** 2) * message_size
        else:
            # 多径衰落模型
            energy = (self.E_elec + self.epsilon_mp * distance ** 4) * message_size
        
        return energy
    
    def calc_receive_energy(self, message_size: int) -> float:
        """
        计算接收能耗
        
        Args:
            message_size: 消息大小 (bits)
            
        Returns:
            接收能耗 (J)
        """
        return self.E_elec * message_size
    
    def calc_aggregation_energy(self, message_size: int) -> float:
        """
        计算数据聚合能耗
        
        Args:
            message_size: 消息大小 (bits)
            
        Returns:
            聚合能耗 (J)
        """
        return self.E_da * message_size
    
    def calc_total_communication_energy(
        self,
        tx_distance: float,
        rx_distance: float,
        message_size: int
    ) -> float:
        """
        计算总通信能耗（发送 + 接收）
        
        Args:
            tx_distance: 发送距离
            rx_distance: 接收距离
            message_size: 消息大小
            
        Returns:
            总能耗
        """
        tx_energy = self.calc_transmit_energy(tx_distance, message_size)
        rx_energy = self.calc_receive_energy(message_size)
        return tx_energy + rx_energy
    
    def get_transmission_mode(self, distance: float) -> str:
        """
        获取传输模式
        
        Args:
            distance: 传输距离
            
        Returns:
            'free_space' 或 'multi_path'
        """
        if distance <= self.d_threshold:
            return 'free_space'
        return 'multi_path'
    
    def estimate_network_energy_budget(
        self,
        n_nodes: int,
        n_rounds: int,
        avg_cluster_size: int,
        message_size: int
    ) -> float:
        """
        估算网络总能耗预算
        
        Args:
            n_nodes: 节点数
            n_rounds: 轮数
            avg_cluster_size: 平均簇大小
            message_size: 消息大小
            
        Returns:
            估算总能耗 (J)
        """
        # 假设最优簇头比例 p = 0.05
        p = 0.05
        n_ch = int(n_nodes * p)
        members_per_ch = n_nodes / n_ch - 1
        
        # 每轮每节点能耗估算
        energy_per_round = 0
        
        for _ in range(n_rounds):
            # 簇头能耗
            # 接收 + 聚合 + 发送
            ch_bits = message_size * (1 + members_per_ch)
            ch_energy = (
                self.E_elec * ch_bits +  # 接收
                self.E_da * ch_bits +     # 聚合
                self.calc_transmit_energy(self.d_threshold, ch_bits)  # 发送
            )
            energy_per_round += n_ch * ch_energy
            
            # 成员节点能耗（仅发送）
            member_bits = message_size
            avg_dist_to_ch = self.d_threshold / 2
            member_energy = self.calc_transmit_energy(avg_dist_to_ch, member_bits)
            energy_per_round += (n_nodes - n_ch) * member_energy
        
        return energy_per_round
