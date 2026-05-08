"""能量模型基类和工厂"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Type, Dict, Any
import numpy as np


class EnergyModel(ABC):
    """能量模型抽象基类"""
    
    @abstractmethod
    def calc_transmit_energy(self, distance: float, message_size: int) -> float:
        """
        计算发送能耗
        
        Args:
            distance: 传输距离 (m)
            message_size: 消息大小 (bits)
            
        Returns:
            发送能耗 (J)
        """
        pass
    
    @abstractmethod
    def calc_receive_energy(self, message_size: int) -> float:
        """
        计算接收能耗
        
        Args:
            message_size: 消息大小 (bits)
            
        Returns:
            接收能耗 (J)
        """
        pass
    
    @abstractmethod
    def get_transmission_mode(self, distance: float) -> str:
        """
        获取传输模式
        
        Args:
            distance: 传输距离
            
        Returns:
            传输模式名称
        """
        pass
    
    @property
    @abstractmethod
    def E_da(self) -> float:
        """数据聚合能耗"""
        pass


class EnergyModelFactory:
    """能量模型工厂"""
    
    _models: Dict[str, Type[EnergyModel]] = {}
    
    @classmethod
    def register(cls, name: str, model_class: Type[EnergyModel]) -> None:
        """
        注册能量模型
        
        Args:
            name: 模型名称
            model_class: 模型类
        """
        if not issubclass(model_class, EnergyModel):
            raise TypeError(f"{model_class} must be subclass of EnergyModel")
        cls._models[name] = model_class
    
    @classmethod
    def create(cls, model_name: str, **kwargs: Any) -> EnergyModel:
        """
        创建能量模型实例
        
        Args:
            model_name: 模型名称
            **kwargs: 模型参数
            
        Returns:
            能量模型实例
        """
        if model_name not in cls._models:
            available = list(cls._models.keys())
            raise ValueError(
                f"Unknown model: {model_name}. Available: {available}"
            )
        return cls._models[model_name](**kwargs)
    
    @classmethod
    def list_models(cls) -> list:
        """列出所有可用的模型"""
        return list(cls._models.keys())


class FirstOrderRadioModel(EnergyModel):
    """
    First Order Radio Model
    
    经典的一阶无线电能量消耗模型。
    
    参数:
        E_elec: 发射/接收电路能耗 (J/bit)
        epsilon_fs: 自由空间能耗系数 (J/bit/m^2)
        epsilon_mp: 多径衰落能耗系数 (J/bit/m^4)
        d_threshold: 距离阈值 (m)
        E_da: 数据聚合能耗 (J/bit)
    """
    
    def __init__(
        self,
        E_elec: float = 50e-9,
        epsilon_fs: float = 10e-12,
        epsilon_mp: float = 0.0013e-12,
        d_threshold: float = 87.0,
        E_da: float = 5e-9
    ):
        self.E_elec = E_elec
        self.epsilon_fs = epsilon_fs
        self.epsilon_mp = epsilon_mp
        self.d_threshold = d_threshold
        self._E_da = E_da
    
    @property
    def E_da(self) -> float:
        """数据聚合能耗"""
        return self._E_da
    
    def calc_transmit_energy(self, distance: float, message_size: int) -> float:
        """计算发送能耗"""
        if distance <= self.d_threshold:
            energy = (self.E_elec + self.epsilon_fs * distance ** 2) * message_size
        else:
            energy = (self.E_elec + self.epsilon_mp * distance ** 4) * message_size
        return energy
    
    def calc_receive_energy(self, message_size: int) -> float:
        """计算接收能耗"""
        return self.E_elec * message_size
    
    def get_transmission_mode(self, distance: float) -> str:
        """获取传输模式"""
        return 'free_space' if distance <= self.d_threshold else 'multi_path'
    
    def calc_aggregation_energy(self, message_size: int) -> float:
        """计算数据聚合能耗"""
        return self._E_da * message_size


class Mica2Model(EnergyModel):
    """
    Mica2 节点能量模型
    
    适用于 Crossbow Mica2 传感节点的能耗模型。
    """
    
    def __init__(
        self,
        Eelec_tx: float = 60e-9,
        Eelec_rx: float = 45e-9,
        Efs: float = 10e-12,
        Emp: float = 0.0013e-12,
        d0: float = 87.0,
        E_da: float = 5e-9
    ):
        self.Eelec_tx = Eelec_tx
        self.Eelec_rx = Eelec_rx
        self.Efs = Efs
        self.Emp = Emp
        self.d0 = d0
        self._E_da = E_da
    
    @property
    def E_da(self) -> float:
        return self._E_da
    
    def calc_transmit_energy(self, distance: float, message_size: int) -> float:
        if distance < self.d0:
            return (self.Eelec_tx + self.Efs * distance**2) * message_size
        return (self.Eelec_tx + self.Emp * distance**4) * message_size
    
    def calc_receive_energy(self, message_size: int) -> float:
        return self.Eelec_rx * message_size
    
    def get_transmission_mode(self, distance: float) -> str:
        return 'free_space' if distance < self.d0 else 'multi_path'


class RssiBasedModel(EnergyModel):
    """
    基于 RSSI 的能量模型
    
    根据接收信号强度指示器 (RSSI) 计算能耗。
    """
    
    def __init__(
        self,
        E_elec: float = 50e-9,
        E_da: float = 5e-9,
        reference_rssi: float = -30.0,
        path_loss_exponent: float = 2.0,
        noise_floor: float = -100.0
    ):
        self.E_elec = E_elec
        self._E_da = E_da
        self.reference_rssi = reference_rssi
        self.path_loss_exponent = path_loss_exponent
        self.noise_floor = noise_floor
        self.reference_distance = 1.0
    
    @property
    def E_da(self) -> float:
        return self._E_da
    
    def _rssi_to_distance(self, rssi: float) -> float:
        """从 RSSI 计算估计距离"""
        if rssi >= self.reference_rssi:
            return self.reference_distance
        
        return self.reference_distance * 10 ** (
            (self.reference_rssi - rssi) / (10 * self.path_loss_exponent)
        )
    
    def calc_transmit_energy(self, distance: float, message_size: int) -> float:
        """基于距离计算发送能耗"""
        tx_power = 1.0
        return (self.E_elec * message_size + tx_power * distance / 1000)
    
    def calc_receive_energy(self, message_size: int) -> float:
        """基于 RSSI 质量计算接收能耗"""
        return self.E_elec * message_size
    
    def get_transmission_mode(self, distance: float) -> str:
        return 'rssi_based'
    
    def estimate_distance_from_rssi(self, rssi: float) -> float:
        """从 RSSI 估算距离"""
        return self._rssi_to_distance(rssi)


class AdaptiveEnergyModel(EnergyModel):
    """
    自适应能量模型
    
    根据网络状态动态调整能量参数。
    """
    
    def __init__(
        self,
        base_model: Optional[EnergyModel] = None,
        energy_factor: float = 1.0,
        adapt_threshold: float = 0.2
    ):
        self.base_model = base_model or FirstOrderRadioModel()
        self.energy_factor = energy_factor
        self.adapt_threshold = adapt_threshold
        self.total_energy_consumed = 0.0
    
    @property
    def E_da(self) -> float:
        return self.base_model.E_da * self.energy_factor
    
    def update_adapt_factor(self, avg_energy_ratio: float) -> None:
        """
        根据平均能量比率更新自适应因子
        
        Args:
            avg_energy_ratio: 网络平均剩余能量比率
        """
        if avg_energy_ratio < self.adapt_threshold:
            self.energy_factor = 0.8
        elif avg_energy_ratio > 0.8:
            self.energy_factor = 1.2
        else:
            self.energy_factor = 1.0
    
    def calc_transmit_energy(self, distance: float, message_size: int) -> float:
        base = self.base_model.calc_transmit_energy(distance, message_size)
        self.total_energy_consumed += base
        return base * self.energy_factor
    
    def calc_receive_energy(self, message_size: int) -> float:
        base = self.base_model.calc_receive_energy(message_size)
        self.total_energy_consumed += base
        return base * self.energy_factor
    
    def get_transmission_mode(self, distance: float) -> str:
        return self.base_model.get_transmission_mode(distance)
    
    def reset(self) -> None:
        """重置累积能耗"""
        self.total_energy_consumed = 0.0


EnergyModelFactory.register('first_order', FirstOrderRadioModel)
EnergyModelFactory.register('mica2', Mica2Model)
EnergyModelFactory.register('rssi', RssiBasedModel)
EnergyModelFactory.register('adaptive', AdaptiveEnergyModel)
