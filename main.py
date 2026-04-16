#!/usr/bin/env python3
"""主入口"""

from src.models.network import Network
from src.energy.radio_model import FirstOrderRadioModel
from src.leach.classic import ClassicLEACH


def main():
    """主函数"""
    print("LEACH Protocol Simulation v1.0.0")
    print("=" * 50)
    
    # 配置
    n_nodes = 100
    area = (0, 100, 0, 100)
    base_station_pos = (50, 50)
    
    # 创建能量模型
    energy_model = FirstOrderRadioModel()
    
    # 创建网络
    network = Network(
        n_nodes=n_nodes,
        area=area,
        base_station_pos=base_station_pos,
        energy_model=energy_model,
        initial_energy=0.5,
        seed=42
    )
    
    print(f"Network: {n_nodes} nodes")
    print(f"Area: {area}")
    print(f"Base Station: {base_station_pos}")
    print()
    
    # 运行仿真
    protocol = ClassicLEACH(p=0.05)
    
    print("Running simulation...")
    results = network.simulate_network(
        rounds=1000,
        protocol_name="leach"
    )
    
    print()
    print("=" * 50)
    print("Results")
    print("=" * 50)
    print(f"Network Lifetime: {results['network_lifetime']} rounds")
    print(f"Half Network Lifetime: {results['half_network_lifetime']} rounds")
    print(f"First Dead Round: {results['first_dead_round']}")
    print(f"Final Energy: {results['final_energy']:.4f} J")


if __name__ == '__main__':
    main()
