"""并行仿真引擎"""

from __future__ import annotations

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
import itertools
import numpy as np

from ..models.network import Network
from ..energy.radio_model import FirstOrderRadioModel
from ..leach.classic import ClassicLEACH


@dataclass
class SimulationConfig:
    """仿真配置"""
    n_nodes: int = 100
    area: Tuple[float, float, float, float] = (0, 100, 0, 100)
    base_station_pos: Tuple[float, float] = (50, 50)
    initial_energy: float = 0.5
    rounds: int = 1000
    protocol_name: str = "leach"
    seed: Optional[int] = None


@dataclass
class SimulationResult:
    """仿真结果"""
    config: SimulationConfig
    run_id: int
    results: Dict[str, Any]
    execution_time: float


class ParallelSimulationEngine:
    """并行仿真引擎"""
    
    def __init__(
        self,
        n_workers: Optional[int] = None,
        use_threads: bool = False
    ):
        """
        初始化并行仿真引擎
        
        Args:
            n_workers: 工作进程数，默认为 CPU 核心数
            use_threads: 是否使用线程（而非进程）
        """
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
        self.use_threads = use_threads
        self._executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    
    def run_parameter_sweep(
        self,
        base_config: SimulationConfig,
        param_grid: Dict[str, List[Any]],
        n_runs: int = 5,
        aggregate: bool = True
    ) -> Dict[str, Any]:
        """
        运行参数扫描实验
        
        Args:
            base_config: 基础配置
            param_grid: 参数网格，如 {'n_nodes': [50, 100, 200]}
            n_runs: 每个参数组合的运行次数
            aggregate: 是否聚合结果
            
        Returns:
            实验结果
        """
        tasks = []
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        for params in itertools.product(*param_values):
            config_dict = {
                'n_nodes': base_config.n_nodes,
                'area': base_config.area,
                'base_station_pos': base_config.base_station_pos,
                'initial_energy': base_config.initial_energy,
                'rounds': base_config.rounds,
                'protocol_name': base_config.protocol_name,
                'seed': base_config.seed,
            }
            
            for name, value in zip(param_names, params):
                config_dict[name] = value
            
            for run_id in range(n_runs):
                config = SimulationConfig(**config_dict)
                if config.seed is not None:
                    config.seed = config.seed + run_id
                tasks.append((config, run_id))
        
        all_results = []
        
        with self._executor_class(max_workers=self.n_workers) as executor:
            futures = {
                executor.submit(
                    _run_single_simulation,
                    config.n_nodes,
                    config.area,
                    config.base_station_pos,
                    config.initial_energy,
                    config.rounds,
                    config.protocol_name,
                    config.seed,
                    run_id
                ): (config, run_id)
                for config, run_id in tasks
            }
            
            for future in as_completed(futures):
                config, run_id = futures[future]
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    print(f"Error running simulation: {e}")
        
        if aggregate:
            return self._aggregate_results(all_results, param_names, param_values)
        
        return {
            'individual_results': [r.__dict__ for r in all_results],
            'param_grid': param_grid,
            'n_runs': n_runs,
        }
    
    def run_parallel_experiments(
        self,
        configs: List[SimulationConfig],
        show_progress: bool = True
    ) -> List[SimulationResult]:
        """
        运行多个配置并行实验
        
        Args:
            configs: 配置列表
            show_progress: 是否显示进度
            
        Returns:
            结果列表
        """
        results = []
        
        with self._executor_class(max_workers=self.n_workers) as executor:
            futures = {}
            
            for i, config in enumerate(configs):
                future = executor.submit(
                    _run_single_simulation,
                    config.n_nodes,
                    config.area,
                    config.base_station_pos,
                    config.initial_energy,
                    config.rounds,
                    config.protocol_name,
                    config.seed,
                    i
                )
                futures[future] = (config, i)
            
            for future in as_completed(futures):
                config, run_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error: {e}")
        
        return results
    
    def _aggregate_results(
        self,
        results: List[SimulationResult],
        param_names: List[str],
        param_values: Tuple[Tuple[Any, ...], ...]
    ) -> Dict[str, Any]:
        """
        聚合结果
        
        Args:
            results: 所有结果
            param_names: 参数名
            param_values: 参数值
            
        Returns:
            聚合后的结果
        """
        aggregated = {
            'summary': {},
            'by_params': {},
            'raw_results': [r.__dict__ for r in results],
        }
        
        for params in itertools.product(*param_values):
            param_key = tuple(zip(param_names, params))
            param_dict = dict(param_key)
            key_str = str(param_key)
            
            matching = [
                r for r in results
                if all(getattr(r.config, name, None) == value 
                       for name, value in param_dict.items())
            ]
            
            if matching:
                lifetimes = [r.results.get('network_lifetime', 0) for r in matching]
                half_lifetimes = [r.results.get('half_network_lifetime', 0) for r in matching]
                
                aggregated['by_params'][key_str] = {
                    'config': param_dict,
                    'n_runs': len(matching),
                    'network_lifetime': {
                        'mean': np.mean(lifetimes),
                        'std': np.std(lifetimes),
                        'min': np.min(lifetimes),
                        'max': np.max(lifetimes),
                    },
                    'half_network_lifetime': {
                        'mean': np.mean(half_lifetimes),
                        'std': np.std(half_lifetimes),
                    },
                    'avg_energy_consumption': np.mean([
                        r.results.get('total_energy', 0) for r in matching
                    ]),
                }
        
        return aggregated


def _run_single_simulation(
    n_nodes: int,
    area: Tuple[float, float, float, float],
    base_station_pos: Tuple[float, float],
    initial_energy: float,
    rounds: int,
    protocol_name: str,
    seed: Optional[int],
    run_id: int
) -> SimulationResult:
    """
    运行单次仿真（独立进程函数）
    
    Args:
        n_nodes: 节点数
        area: 区域
        base_station_pos: 基站位置
        initial_energy: 初始能量
        rounds: 轮数
        protocol_name: 协议名
        seed: 随机种子
        run_id: 运行 ID
        
    Returns:
        仿真结果
    """
    import time
    
    config = SimulationConfig(
        n_nodes=n_nodes,
        area=area,
        base_station_pos=base_station_pos,
        initial_energy=initial_energy,
        rounds=rounds,
        protocol_name=protocol_name,
        seed=seed
    )
    
    start_time = time.time()
    
    energy_model = FirstOrderRadioModel()
    
    network = Network(
        n_nodes=n_nodes,
        area=area,
        base_station_pos=base_station_pos,
        energy_model=energy_model,
        initial_energy=initial_energy,
        seed=seed
    )
    
    results = network.simulate_network(
        rounds=rounds,
        protocol_name=protocol_name
    )
    
    execution_time = time.time() - start_time
    
    return SimulationResult(
        config=config,
        run_id=run_id,
        results=results,
        execution_time=execution_time
    )


class BatchSimulator:
    """批量仿真器"""
    
    def __init__(self, engine: Optional[ParallelSimulationEngine] = None):
        """初始化批量仿真器"""
        self.engine = engine or ParallelSimulationEngine()
        self._results_cache: Dict[str, SimulationResult] = {}
    
    def compare_protocols(
        self,
        n_nodes: int = 100,
        area: Tuple[float, float, float, float] = (0, 100, 0, 100),
        rounds: int = 1000,
        protocols: Optional[List[str]] = None,
        n_runs: int = 10
    ) -> Dict[str, Any]:
        """
        对比不同协议
        
        Args:
            n_nodes: 节点数
            area: 区域
            rounds: 轮数
            protocols: 协议列表
            n_runs: 每个协议运行次数
            
        Returns:
            对比结果
        """
        protocols = protocols or ['leach', 'leach_c', 'leach_ee', 'leach_m']
        
        comparison = {
            'protocols': {},
            'summary': {}
        }
        
        for protocol in protocols:
            config = SimulationConfig(
                n_nodes=n_nodes,
                area=area,
                rounds=rounds,
                protocol_name=protocol,
                seed=42
            )
            
            results = self.engine.run_parameter_sweep(
                base_config=config,
                param_grid={},
                n_runs=n_runs,
                aggregate=True
            )
            
            comparison['protocols'][protocol] = results
        
        all_lifetimes = {
            p: comparison['protocols'][p]['summary'].get('network_lifetime', 0)
            for p in protocols
        }
        
        comparison['summary'] = {
            'best_protocol': max(all_lifetimes, key=all_lifetimes.get),
            'lifetimes': all_lifetimes,
        }
        
        return comparison
    
    def save_results(
        self,
        results: Dict[str, Any],
        output_path: str
    ) -> None:
        """
        保存结果到文件
        
        Args:
            results: 结果字典
            output_path: 输出路径
        """
        import json
        from pathlib import Path
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def load_results(self, input_path: str) -> Dict[str, Any]:
        """
        从文件加载结果
        
        Args:
            input_path: 输入路径
            
        Returns:
            结果字典
        """
        import json
        
        with open(input_path, 'r') as f:
            return json.load(f)
