#!/usr/bin/env python3
"""
Protocol-algorithm v2.0 - Python 示例脚本

演示如何使用 Python API 运行 LEACH 协议仿真
"""

import sys
sys.path.insert(0, '../python')

try:
    import protocol_algo as pa
    print("✅ 成功导入 protocol_algo 模块")
except ImportError as e:
    print(f"⚠️  模块未安装，使用模拟数据演示: {e}")
    
    # 模拟类用于演示
    class Network:
        def __init__(self, nodes=100, area=100.0, base_station=(50.0, 150.0)):
            self.nodes = nodes
            self.area = area
            self.base_station = base_station
        
        def __repr__(self):
            return f"Network(nodes={self.nodes}, area={self.area})"
    
    class LEACH:
        def __init__(self, p=0.05, rounds=100, initial_energy=0.5, seed=42):
            self.p = p
            self.rounds = rounds
            self.initial_energy = initial_energy
            self.seed = seed
        
        def run(self, network):
            return SimulationResult(self.rounds, network.nodes)
    
    class SimulationResult:
        def __init__(self, rounds, initial_nodes):
            self.rounds = rounds
            self.initial_nodes = initial_nodes
            self.final_alive = int(initial_nodes * 0.7)
        
        def survival_rate(self):
            return (self.final_alive / self.initial_nodes) * 100
    
    class Visualizer:
        def __init__(self, style="modern"):
            self.style = style
        
        def plot_network(self, network, result):
            print(f"  📊 绘制网络图 (风格：{self.style})")
        
        def plot_metrics(self, result):
            print(f"  📈 绘制指标图表")
        
        def save(self, path):
            print(f"  💾 保存至：{path}")

def main():
    print("=" * 60)
    print("Protocol-algorithm v2.0 - Python 示例")
    print("=" * 60)
    
    # 创建网络
    print("\n1️⃣  创建网络配置...")
    network = pa.Network(
        nodes=100,
        area=100.0,
        base_station=(50.0, 150.0)
    )
    print(f"   {network}")
    
    # 配置 LEACH
    print("\n2️⃣  配置 LEACH 协议...")
    leach = pa.LEACH(
        p=0.05,          # 5% 簇头概率
        rounds=100,      # 100 轮仿真
        initial_energy=0.5,  # 初始能量 0.5J
        seed=42          # 随机种子
    )
    print(f"   {leach}")
    
    # 运行仿真
    print("\n3️⃣  运行仿真...")
    result = leach.run(network)
    print(f"   {result}")
    print(f"   存活率：{result.survival_rate():.1f}%")
    
    # 可视化
    print("\n4️⃣  可视化结果...")
    viz = pa.Visualizer(style="modern")
    print(f"   {viz}")
    viz.plot_network(network, result)
    viz.plot_metrics(result)
    viz.save("output.png")
    
    print("\n" + "=" * 60)
    print("✅ 示例执行完成！")
    print("=" * 60)
    
    # 更多示例
    print("\n💡 更多用法:")
    print("""
# 不同节点数量
for n in [50, 100, 200, 500]:
    network = Network(nodes=n)
    result = leach.run(network)
    print(f"{n} 节点 - 存活率：{result.survival_rate():.1f}%")

# 不同簇头概率
for p in [0.03, 0.05, 0.08, 0.1]:
    leach = LEACH(p=p, rounds=100)
    result = leach.run(network)
    print(f"p={p:.2f} - 存活率：{result.survival_rate():.1f}%")

# 可视化保存
viz = Visualizer(style="dark")
viz.save("network_dark.png")
    """)

if __name__ == "__main__":
    main()
