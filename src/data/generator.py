"""数据生成器"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

from ..models.network import Network
from ..energy.radio_model import FirstOrderRadioModel
from ..leach.classic import ClassicLEACH


class DataGenerator:
    """训练数据生成器"""
    
    def __init__(
        self,
        n_nodes: int = 100,
        area: Tuple[float, float, float, float] = (0, 100, 0, 100),
        base_station_pos: Tuple[float, float] = (50, 50),
        initial_energy: float = 0.5
    ):
        """
        初始化
        
        Args:
            n_nodes: 节点数量
            area: 区域范围
            base_station_pos: 基站位置
            initial_energy: 初始能量
        """
        self.n_nodes = n_nodes
        self.area = area
        self.base_station_pos = base_station_pos
        self.initial_energy = initial_energy
    
    def generate_training_data(
        self,
        n_rounds: int = 500,
        protocol_name: str = "leach",
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        生成训练数据
        
        Args:
            n_rounds: 模拟轮数
            protocol_name: 协议名称
            seed: 随机种子
            
        Returns:
            训练数据 DataFrame
        """
        if seed is not None:
            np.random.seed(seed)
        
        # 创建网络
        energy_model = FirstOrderRadioModel()
        network = Network(
            n_nodes=self.n_nodes,
            area=self.area,
            base_station_pos=self.base_station_pos,
            energy_model=energy_model,
            initial_energy=self.initial_energy,
            seed=seed
        )
        
        # 收集数据
        records = []
        
        for round_num in range(n_rounds):
            network.simulate_round(protocol_name)
            
            for node in network.nodes:
                bs_pos = network.base_station.position
                
                record = {
                    'round': round_num,
                    'node_id': node.id,
                    'x': node.x,
                    'y': node.y,
                    'energy': node.energy,
                    'energy_ratio': node.energy / node.initial_energy,
                    'dist_to_bs': node.distance_to(bs_pos),
                    'is_alive': node.is_alive,
                    'is_cluster_head': node.is_cluster_head,
                    'round_dead': node.round_dead,
                }
                
                # 添加邻居密度
                neighbors = network.get_neighbors(node, 30.0)
                record['n_neighbors'] = len(neighbors)
                
                records.append(record)
        
        df = pd.DataFrame(records)
        
        return df
    
    def generate_balanced_dataset(
        self,
        n_rounds: int = 500,
        positive_ratio: float = 0.1,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成平衡数据集
        
        Args:
            n_rounds: 模拟轮数
            positive_ratio: 正例比例
            seed: 随机种子
            
        Returns:
            X, y
        """
        df = self.generate_training_data(n_rounds, seed=seed)
        
        # 过滤存活节点
        df = df[df['is_alive']]
        
        # 特征列
        feature_cols = [
            'x', 'y', 'energy', 'energy_ratio',
            'dist_to_bs', 'n_neighbors'
        ]
        
        X = df[feature_cols].values
        
        # 平衡采样
        ch_mask = df['is_cluster_head']
        
        ch_samples = df[ch_mask]
        non_ch_samples = df[~ch_mask]
        
        # 调整比例
        n_positive = int(len(non_ch_samples) * positive_ratio)
        n_positive = min(n_positive, len(ch_samples))
        
        if len(ch_samples) > n_positive:
            ch_samples = ch_samples.sample(n=n_positive, random_state=seed)
        
        n_negative = int(n_positive / positive_ratio) - n_positive
        if len(non_ch_samples) > n_negative:
            non_ch_samples = non_ch_samples.sample(n=n_negative, random_state=seed)
        
        # 合并
        balanced_df = pd.concat([ch_samples, non_ch_samples])
        
        X = balanced_df[feature_cols].values
        y = balanced_df['is_cluster_head'].astype(int).values
        
        return X, y
    
    def save_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        path: str,
        feature_names: Optional[List[str]] = None
    ):
        """
        保存数据集
        
        Args:
            X: 特征
            y: 标签
            path: 保存路径
            feature_names: 特征名
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        df = pd.DataFrame(X, columns=feature_names)
        df['label'] = y
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        
        print(f"Dataset saved to {path}")
