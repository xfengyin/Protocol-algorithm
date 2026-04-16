"""命令行接口"""

import argparse
import sys
from pathlib import Path

import yaml

from src.models.network import Network
from src.energy.radio_model import FirstOrderRadioModel
from src.simulation.engine import SimulationEngine
from src.visualization.metrics_plots import MetricsPlotter


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description='LEACH Protocol Simulation'
    )
    
    parser.add_argument(
        '--config',
        '-c',
        type=str,
        help='Config file path'
    )
    
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default='./output',
        help='Output directory'
    )
    
    parser.add_argument(
        '--protocol',
        '-p',
        type=str,
        default='leach',
        choices=['leach', 'leach-c', 'leach-ee', 'leach-m', 'leach_ai'],
        help='Protocol name'
    )
    
    parser.add_argument(
        '--rounds',
        '-r',
        type=int,
        default=1000,
        help='Number of rounds'
    )
    
    parser.add_argument(
        '--nodes',
        '-n',
        type=int,
        default=100,
        help='Number of nodes'
    )
    
    parser.add_argument(
        '--seed',
        '-s',
        type=int,
        help='Random seed'
    )
    
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip plotting'
    )
    
    args = parser.parse_args()
    
    # 加载配置或使用默认值
    if args.config:
        config = load_config(args.config)
    else:
        config = {
            'network': {
                'n_nodes': args.nodes,
                'area': (0, 100, 0, 100),
                'base_station_pos': (50, 50),
                'initial_energy': 0.5,
            },
            'energy_model': {},
            'simulation': {
                'rounds': args.rounds,
            },
            'protocol': args.protocol,
        }
    
    print("=" * 60)
    print("LEACH Protocol Simulation")
    print("=" * 60)
    print(f"Protocol: {config.get('protocol', args.protocol)}")
    print(f"Nodes: {config['network']['n_nodes']}")
    print(f"Rounds: {config['simulation']['rounds']}")
    print()
    
    # 创建网络
    network_config = config['network']
    energy_model = FirstOrderRadioModel(**config.get('energy_model', {}))
    
    network = Network(
        n_nodes=network_config['n_nodes'],
        area=tuple(network_config['area']),
        base_station_pos=tuple(network_config['base_station_pos']),
        energy_model=energy_model,
        initial_energy=network_config.get('initial_energy', 0.5),
        seed=args.seed or config.get('seed')
    )
    
    # 创建仿真引擎
    engine = SimulationEngine(network)
    
    # 运行仿真
    print("Running simulation...")
    results = engine.simulate(
        rounds=config['simulation']['rounds'],
        protocol_name=config.get('protocol', args.protocol)
    )
    
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Total Rounds: {results.get('total_rounds_simulated', len(results['rounds']))}")
    print(f"First Node Dead: Round {results.get('first_dead_round', 'N/A')}")
    print(f"Half Network Dead: Round {results.get('half_dead_round', 'N/A')}")
    print(f"Final Alive: {results['alive_nodes'][-1]}")
    print()
    
    # 保存结果
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(output_dir / 'results.json', 'w') as f:
        # 转换结果
        results_to_save = {
            'rounds': results['rounds'],
            'alive_nodes': results['alive_nodes'],
            'dead_nodes': results['dead_nodes'],
            'total_energy': [float(e) for e in results['total_energy']],
            'cluster_heads': results['cluster_heads'],
            'first_dead_round': results.get('first_dead_round'),
            'half_dead_round': results.get('half_dead_round'),
            'total_rounds_simulated': results.get('total_rounds_simulated'),
        }
        json.dump(results_to_save, f, indent=2)
    
    print(f"Results saved to {output_dir / 'results.json'}")
    
    # 绘图
    if not args.no_plot:
        print("Generating plots...")
        plotter = MetricsPlotter()
        plotter.plot_all_metrics(results, str(output_dir))
        print(f"Plots saved to {output_dir}")


if __name__ == '__main__':
    main()
