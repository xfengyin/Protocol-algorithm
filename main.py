#!/usr/bin/env python3
"""主入口"""

from src.energy.radio_model import FirstOrderRadioModel
from src.models.network import Network


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
        seed=42,
    )

    print(f"Network: {n_nodes} nodes")
    print(f"Area: {area}")
    print(f"Base Station: {base_station_pos}")
    print()

    # 运行仿真（simulate_network 按 protocol_name 从协议注册表取实现，
    # 所以这里不需要（也不该）先造一个协议实例）
    print("Running simulation...")
    results = network.simulate_network(rounds=1000, protocol_name="leach")

    print()
    print("=" * 50)
    print("Results")
    print("=" * 50)
    print(f"Network Lifetime: {results['network_lifetime']} rounds")
    print(f"Half Network Lifetime: {results['half_network_lifetime']} rounds")
    print(f"First Dead Round: {results['first_dead_round']}")
    print(f"Final Energy: {results['final_energy']:.4f} J")


if __name__ == "__main__":
    main()
