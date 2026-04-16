"""仿真引擎"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass, field
from enum import Enum

from ..models.network import Network, NetworkMetrics
from ..leach.variants import LEACHRegistry


@dataclass
class SimulationEvent:
    """仿真事件"""
    round_number: int
    event_type: str
    data: Dict[str, Any] = field(default_factory=dict)


class SimulationEngine:
    """
    事件驱动仿真引擎
    
    支持:
    - 事件队列
    - 回调钩子
    - 并行实验
    """
    
    def __init__(self, network: Network):
        """
        初始化
        
        Args:
            network: 网络对象
        """
        self.network = network
        self.event_queue: List[SimulationEvent] = []
        self.callbacks: Dict[str, List[Callable]] = {
            'on_round_start': [],
            'on_round_end': [],
            'on_node_dead': [],
            'on_cluster_head_selected': [],
            'on_simulation_end': [],
        }
        
        self.running = False
    
    def register_callback(self, event_type: str, callback: Callable):
        """
        注册回调
        
        Args:
            event_type: 事件类型
            callback: 回调函数
        """
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
    
    def emit_event(self, event: SimulationEvent):
        """触发事件"""
        self.event_queue.append(event)
        
        if event.event_type in self.callbacks:
            for callback in self.callbacks[event.event_type]:
                callback(event)
    
    def simulate_round(
        self,
        protocol_name: str = "leach",
        **kwargs
    ) -> NetworkMetrics:
        """
        模拟一轮
        
        Args:
            protocol_name: 协议名称
            **kwargs: 协议参数
            
        Returns:
            网络指标
        """
        # 触发开始事件
        self.emit_event(SimulationEvent(
            round_number=self.network.current_round,
            event_type='on_round_start',
            data={'alive_nodes': self.network.n_alive}
        ))
        
        # 执行仿真
        metrics = self.network.simulate_round(protocol_name, **kwargs)
        
        # 检查死亡节点
        for node in self.network.dead_nodes:
            if node.round_dead is None:
                node.round_dead = self.network.current_round
                self.emit_event(SimulationEvent(
                    round_number=self.network.current_round,
                    event_type='on_node_dead',
                    data={'node_id': node.id}
                ))
        
        # 触发结束事件
        self.emit_event(SimulationEvent(
            round_number=self.network.current_round,
            event_type='on_round_end',
            data={'metrics': metrics}
        ))
        
        return metrics
    
    def simulate(
        self,
        rounds: int,
        protocol_name: str = "leach",
        stop_condition: Optional[Callable] = None,
        progress_callback: Optional[Callable] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        运行仿真
        
        Args:
            rounds: 最大轮数
            protocol_name: 协议名称
            stop_condition: 停止条件
            progress_callback: 进度回调
            **kwargs: 协议参数
            
        Returns:
            仿真结果
        """
        self.running = True
        results = {
            'rounds': [],
            'alive_nodes': [],
            'metrics': [],
        }
        
        try:
            for r in range(rounds):
                if not self.running:
                    break
                
                metrics = self.simulate_round(protocol_name, **kwargs)
                
                results['rounds'].append(r)
                results['alive_nodes'].append(metrics.alive_nodes)
                results['metrics'].append(metrics)
                
                if progress_callback:
                    progress_callback(r, metrics)
                
                if stop_condition and stop_condition(self.network, r):
                    break
                    
        finally:
            self.running = False
            self.emit_event(SimulationEvent(
                round_number=r,
                event_type='on_simulation_end',
                data={'total_rounds': len(results['rounds'])}
            ))
        
        return results
    
    def stop(self):
        """停止仿真"""
        self.running = False


class ParallelExperimentRunner:
    """并行实验运行器"""
    
    def __init__(self, n_workers: int = 4):
        """
        初始化
        
        Args:
            n_workers: 并行工作进程数
        """
        self.n_workers = n_workers
    
    def run_single(
        self,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        运行单次实验
        
        Args:
            config: 实验配置
            
        Returns:
            实验结果
        """
        from ..models.network import Network
        from ..energy.radio_model import FirstOrderRadioModel
        
        # 创建网络
        energy_model = FirstOrderRadioModel(**config.get('energy_model', {}))
        network = Network(
            n_nodes=config['n_nodes'],
            area=config['area'],
            base_station_pos=config['base_station_pos'],
            energy_model=energy_model,
            initial_energy=config.get('initial_energy', 0.5),
            seed=config.get('seed')
        )
        
        # 创建引擎
        engine = SimulationEngine(network)
        
        # 运行
        results = engine.simulate(
            rounds=config['rounds'],
            protocol_name=config['protocol']
        )
        
        return {
            'config': config,
            'results': results,
            'summary': self._summarize(results)
        }
    
    def run_multiple(
        self,
        configs: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        运行多次实验
        
        Args:
            configs: 实验配置列表
            show_progress: 是否显示进度
            
        Returns:
            实验结果列表
        """
        results = []
        
        for i, config in enumerate(configs):
            if show_progress:
                print(f"Running experiment {i+1}/{len(configs)}: {config['protocol']}")
            
            result = self.run_single(config)
            results.append(result)
        
        return results
    
    def _summarize(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """汇总结果"""
        alive_history = np.array(results['alive_nodes'])
        
        first_dead = np.argmax(alive_history < len(alive_history))
        if alive_history[first_dead] == len(alive_history):
            first_dead = len(alive_history)
        
        half_dead = np.argmax(alive_history <= len(alive_history) / 2)
        if alive_history[half_dead] > len(alive_history) / 2:
            half_dead = len(alive_history)
        
        return {
            'total_rounds': len(alive_history),
            'first_dead_round': int(first_dead),
            'half_dead_round': int(half_dead),
            'final_alive': int(alive_history[-1]),
            'energy_stability': float(np.std([m.total_energy for m in results['metrics']])),
        }
